# import e2cnn.group.groups

import torch

# --- Monkey-patch Tensor.__setitem__ to accept uint8 masks -------------
_orig_setitem = torch.Tensor.__setitem__

def _safe_setitem(self, key, value):
    # intercept "tensor[mask] = …" where mask is uint8
    if isinstance(key, torch.Tensor) and key.dtype is torch.uint8:
        key = key.to(torch.bool)
    return _orig_setitem(self, key, value)

torch.Tensor.__setitem__ = _safe_setitem

import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Type, Callable, Literal, Optional
from torch import einsum
from torch.nn import functional as F
# import logging


# import e2cnn
# import e2cnn.nn as e2nn
# from e2cnn.group import directsum

import escnn.nn as e2nn
import escnn

from multiplex_model.modules import Hyperkernel, MultiplexImageDecoder

import torch
import torch.nn as nn
import escnn.nn as e2nn


class EquivariantPixelLN(nn.Module):
    """
    Equivariant per-pixel normalization for BL-regular FieldTypes.

    For each spatial position independently (no batch dependency):
    1. **Center** the scalar (frequency-0) sub-component of each BL-regular
       field by subtracting the mean across all such scalar components.
       Higher-frequency components are left untouched (equivariant).
    2. **Scale** all fields by the mean per-field L2 magnitude (RMSNorm-style).
    3. **Affine**: learned per-field gamma (all fields) and per-field beta
       (scalar sub-component only — equivariant).

    Fully vectorized — no Python loops over fields in the forward pass.
    """

    def __init__(
        self,
        in_type: e2nn.FieldType,
        eps: float = 1e-6,
        center_scalar: bool = True,
        affine: bool = True,
    ):
        super().__init__()
        self.in_type = in_type
        self.out_type = in_type
        self.eps = eps
        self.center_scalar = center_scalar
        self.affine = affine

        # All fields must have the same representation size (BL-regular copies)
        sizes = [r.size for r in in_type.representations]
        assert len(set(sizes)) == 1, (
            f"EquivariantPixelLN requires uniform field size, got {set(sizes)}"
        )
        self.repr_dim = sizes[0]  # e.g. 3 for max_freq=1
        self.n_fields = len(in_type.representations)
        total_channels = self.n_fields * self.repr_dim

        # field_indices: maps each channel to its field index
        # e.g. for repr_dim=3, n_fields=4: [0,0,0, 1,1,1, 2,2,2, 3,3,3]
        field_indices = torch.arange(self.n_fields).repeat_interleave(self.repr_dim)
        self.register_buffer("field_indices", field_indices)  # [total_channels]

        # scalar_mask: True for the first (frequency-0) component of each field
        scalar_mask = torch.zeros(total_channels, dtype=torch.bool)
        scalar_mask[::self.repr_dim] = True
        self.register_buffer("scalar_mask", scalar_mask)  # [total_channels]

        # Indices of scalar channels for gather/scatter
        scalar_indices = torch.where(scalar_mask)[0]
        self.register_buffer("scalar_indices", scalar_indices)  # [n_fields]

        if affine:
            self.gamma = nn.Parameter(torch.ones(self.n_fields))
            if center_scalar:
                self.beta = nn.Parameter(torch.zeros(self.n_fields))
            else:
                self.beta = None

    def forward(self, x: e2nn.GeometricTensor) -> e2nn.GeometricTensor:
        t = x.tensor  # [B, C, H, W]
        B, C, H, W = t.shape

        # ── 1) Center scalar sub-components ──────────────────────────────
        if self.center_scalar:
            # Extract scalar (freq-0) channels: [B, n_fields, H, W]
            scalars = t[:, self.scalar_mask, :, :]
            scalar_mean = scalars.mean(dim=1, keepdim=True)  # [B, 1, H, W]

            # Subtract mean only from scalar channels
            t = t.clone()
            t[:, self.scalar_mask, :, :] = scalars - scalar_mean

        # ── 2) Per-pixel scale: mean of per-field L2 magnitudes ──────────
        # Reshape to [B, n_fields, repr_dim, H, W], compute L2 per field
        t_fields = t.view(B, self.n_fields, self.repr_dim, H, W)
        magnitudes = torch.linalg.vector_norm(
            t_fields, ord=2, dim=2,
        )  # [B, n_fields, H, W]

        # s(b,h,w) = mean_f ||field_f||_2
        s = magnitudes.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        denom = s + self.eps  # [B, 1, H, W]

        # Rescale: broadcast denom over (n_fields, repr_dim)
        out = t_fields / denom.unsqueeze(2)  # [B, n_fields, repr_dim, H, W]

        # ── 3) Affine ────────────────────────────────────────────────────
        if self.affine:
            # gamma per field, broadcast over repr_dim and spatial dims
            gamma = self.gamma.view(1, self.n_fields, 1, 1, 1)
            out = out * gamma

            if self.beta is not None:
                # beta only on scalar (freq-0) component — index 0 of repr_dim
                beta = self.beta.view(1, self.n_fields, 1, 1)
                out[:, :, 0, :, :] = out[:, :, 0, :, :] + beta

        out = out.reshape(B, C, H, W)
        return e2nn.GeometricTensor(out, self.out_type)


class GRNByIrrep(nn.Module):
    """
    One γ, β per *copy* of an irrep (no matter its dimension).
    This keeps equivariance for any ESCNN FieldType.
    """

    def __init__(self, field_type: e2nn.FieldType, eps=1e-6):
        super().__init__()
        self.eps = eps

        # Build a list:  [(slice, dim)]     one entry per field
        # `slice`   → the channel range in the tensor
        # `dim`     → irrep dimension (1 for trivial, 2 for a real harmonic, ...)
        self.fields = []
        start = 0
        for r in field_type:
            size = r.size
            self.fields.append((slice(start, start+size), size))
            start += size
        self.n_fields = len(self.fields)

        field_indices = torch.empty(start, dtype=torch.long)
        for idx, (sl, _) in enumerate(self.fields):
            field_indices[sl] = idx
        self.register_buffer("field_indices", field_indices)

        # One γ, β per field copy
        self.gamma = nn.Parameter(torch.zeros(1, len(self.fields), 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, len(self.fields), 1, 1))
        self.ftype = field_type

    # def forward(self, x: e2nn.GeometricTensor):
    #     t = x.tensor                                    # [B,C,H,W]
    #     B, C, H, W = t.shape

    #     # compute per-field L2 norm  → shape  [B, n_fields, 1, 1]
    #     norms = []
    #     for sl, d in self.fields:
    #         # sum squares over irrep dimension `d` and spatial dims H,W
    #         n = t[:, sl].reshape(B, d, -1)              # [B,d,H*W]
    #         n = torch.linalg.norm(n, ord=2, dim=(1,2), keepdim=True)
    #         norms.append(n)
    #     gx = torch.cat(norms, dim=1)                    # [B,nf,1,1]

    #     # same rescaling as in the paper
    #     nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)   # [B,nf,1,1]

    #     # broadcast γ, β to the corresponding channels
    #     out = t.clone()
    #     # for i, (sl, _) in enumerate(self.fields):
    #     #     out[:, sl] = (self.gamma[:, i] * (t[:, sl] * nx[:, i])
    #     #                   + self.beta[:, i] + t[:, sl])
    #     for i, (sl, _) in enumerate(self.fields):
    #         gamma_i = self.gamma[:, i:i+1, ...]      # (1,1,1,1)
    #         beta_i  = self.beta[:,  i:i+1, ...]      # (1,1,1,1)
    #         nx_i    = nx[:,   i:i+1, ...]            # (B,1,1,1)

    #         out[:, sl] = gamma_i * (t[:, sl] * nx_i) + beta_i + t[:, sl]
    #     return e2nn.GeometricTensor(out, self.ftype)
    def forward(self, x: e2nn.GeometricTensor):
        t = x.tensor                                # [B, C, H, W]
        B, C, H, W = t.shape

        # ----------------------------------------------------------
        # 1) per-field L2 norm  →  [B, n_fields, 1, 1]
        # ----------------------------------------------------------
        field_idx = self.field_indices.view(1, C)    # [1, C]
        channel_sq = t.reshape(B, C, -1).square().sum(dim=2)  # [B, C]
        gx = channel_sq.new_zeros(B, self.n_fields)  # match dtype to source
        gx.scatter_add_(1, field_idx.expand(B, -1), channel_sq)
        gx = torch.sqrt(gx).view(B, self.n_fields, 1, 1)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)

        # ----------------------------------------------------------
        # 2) apply γ, β per field (keep axis 1!)
        # ----------------------------------------------------------
        field_idx_c = self.field_indices.view(1, C, 1, 1).expand(B, -1, -1, -1)
        gamma_expanded = self.gamma.expand(B, -1, -1, -1)
        beta_expanded = self.beta.expand(B, -1, -1, -1)
        gamma_per_channel = torch.gather(gamma_expanded, 1, field_idx_c)
        beta_per_channel = torch.gather(beta_expanded, 1, field_idx_c)
        nx_per_channel = torch.gather(nx, 1, field_idx_c)

        out = t + gamma_per_channel * (t * nx_per_channel) + beta_per_channel

        return e2nn.GeometricTensor(out, self.ftype)


# logging.basicConfig(level=logging.DEBUG)

# class EquivariantMultiplexAutoencoderOld(nn.Module):
#     """Multiplex image Transformer with Superkernel and Multiplex Image Decoder."""

#     def __init__(
#             self, 
#             num_channels: int,
#             input_image_size: int,
#             superkernel_embedding_dim: int,
#             superkernel_depth: int,
#             superkernel_heads: int,
#             superkernel_layer_type: Literal['conv', 'linear'],
#             encoder_config: Dict,
#             decoder_config: Dict,
#             superkernel_kernel_size: int = None,
#             superkernel_conv_padding: int = None,
#             superkernel_conv_stride: int = 1,
#             mlp_ratio: float = 4.,
#             **kwargs
#             ):
#         """Initialize the Multiplex Transformer model.

#         Args:
#             num_channels (int): Number of channels/markers in the dataset.
#             input_image_size (int): Size of the input image.
#             superkernel_embedding_dim (int): Embedding dimension for the Superkernel.
#             superkernel_depth (int): Number of layers in the Superkernel model.
#             superkernel_heads (int): Number of heads per channel embedding in the Superkernel model.
#             superkernel_layer_type (Literal['conv', 'linear']): Type of the Superkernel layer.
#             encoder_config (Dict): Configuration for the encoder.
#             decoder_config (Dict): Configuration for the decoder.
#             superkernel_kernel_size (int, optional): Size of Superkernel kernel if conv type. Defaults to None.
#             superkernel_conv_padding (int, optional): Convolution padding if conv type. Defaults to None.
#             superkernel_conv_stride (int, optional): Convolution stride if conv type. Defaults to 1.
#             mlp_ratio (float, optional): MLP ratio. Defaults to 4..
#         """
#         super().__init__()
#         self.num_channels = num_channels
#         self.input_image_size = input_image_size
#         self.superkernel_embedding_dim = superkernel_embedding_dim
#         self.superkernel_depth = superkernel_depth
#         self.superkernel_heads = superkernel_heads
#         self.superkernel_layer_type = superkernel_layer_type
#         self.superkernel_kernel_size = superkernel_kernel_size
#         self.superkernel_conv_padding = superkernel_conv_padding
#         self.superkernel_conv_stride = superkernel_conv_stride
#         self.mlp_ratio = mlp_ratio


#         self.superkernel = Superkernel(
#             num_channels=num_channels, 
#             embedding_dim=superkernel_embedding_dim, 
#             num_layers=superkernel_depth, 
#             num_heads=superkernel_heads, 
#             mlp_ratio=mlp_ratio, 
#             layer_type=superkernel_layer_type,
#             kernel_size=superkernel_kernel_size,
#             **kwargs
#         )
#         self.act = nn.GELU()

#         self.encoder = EscnnMultiplexImageEncoder(
#             **encoder_config
#         )

#         self.decoder = MultiplexImageDecoder(
#             **decoder_config
#         )

#     def forward(
#             self, 
#             x: torch.Tensor, 
#             encoded_indices: torch.Tensor, 
#             decoded_indices: torch.Tensor
#         ) -> torch.Tensor:
#         # print("shape", x.shape)
#         B = x.shape[0]
#         # print(f'Input shape: {x.shape}, Encoded indices shape: {encoded_indices.shape}, Decoded indices shape: {decoded_indices.shape}')
#         # print((f"Input isnan: {torch.isnan(x).any()}, "))
#         # print((f"Encoded indices isnan: {torch.isnan(encoded_indices).any()}, "))
#         # print((f"Decoded indices isnan: {torch.isnan(decoded_indices).any()}, "))
#         superkernel_weights = self.superkernel(encoded_indices)


#         # print((f"Superkernel weights isnan: {torch.isnan(superkernel_weights).any()}, "))
#         if self.superkernel_layer_type == 'conv':
#             x = torch.cat([
#                 F.conv2d(
#                     x[i].unsqueeze(0), 
#                     superkernel_weights[i].to(x.dtype), 
#                     padding=self.superkernel_conv_padding,
#                     stride=self.superkernel_conv_stride
#                 )
#                 for i in range(B)
#             ])
            
#         else:
#             x = torch.einsum('bchw, bce -> behw', x, superkernel_weights.to(x.dtype))

#         # print(f'After superkernel shape: {x.shape}')
#         # print((f"After superkernel isnan: {torch.isnan(x).any()}, "))
#         x = self.act(x)

#         # print(f'After activation shape: {x.shape}')
#         # print((f"After activation isnan: {torch.isnan(x).any()}, "))
#         x = self.encoder(x)
#         # print(f"latent isnan: {torch.isnan(x).any()}")

#         # print(f'After encoder shape: {x.shape}')
#         # print((f"After encoder isnan: {torch.isnan(x).any()}, "))
#         latent = x
#         # print(f'Latent shape: {latent.shape}')
#         # latent, features = x[:, 0], x[:, 1:]
#         # latent = x.mean(dim=(2, 3))
#         # x = features.permute(0, 2, 1).reshape(B, 768, 14, 14)

#         x = self.decoder(x, decoded_indices)
#         # x = x[:, :, ]
        
#         # print(f"decoded isnan: {torch.isnan(x).any()}")
#         # print(f'After decoder shape: {x.shape}')
#         # print((f"After decoder isnan: {torch.isnan(x).any()}, "))        
#         return x, latent


class EquivariantHyperkernel(nn.Module):
    """Marker-conditioned channel mixer acting field-wise on BL-regular features.

    Two parameterizations of the marker-conditioned linear map between
    BL-regular field types are supported, both equivariant by construction:

    * ``intertwiner_basis="scalar"`` (cheapest): one real weight per
      (input field, output field) pair per marker. Acts as ``c · I_R`` on each
      input field — i.e. uniform scaling. Strict subset of the full intertwiner
      space; cannot independently scale different irrep sub-components.

    * ``intertwiner_basis="full"`` (full equivariant expressivity under O(2)):
      Per (input field, output field) pair, learns ``Σ_irrep_type m_t²`` real
      weights, where the sum runs over distinct irrep types in the BL-regular
      field and ``m_t`` is the multiplicity of irrep type ``t`` within one
      field. Each (mult_out × mult_in) block of weights mixes copies of the
      same irrep type, while leaving the irrep-dim axis untouched. For O(2)
      BL-regular(M), this is ``2 + 4·M`` parameters per pair (1 trivial,
      1 sign, and 4 per non-trivial frequency from the 2-copy multiplicity).
      Using End(irrep) = ℝ (true for finite groups and O(2) — uses
      ``flipRot2dOnR2``); SO(2) would have End(freq_k) = ℂ which is not
      handled here.

    In both cases the per-irrep scalar commutes with the corresponding
    irrep representation matrix, so equivariance is preserved.

    Input  : plain tensor (B, C * input_fields * repr_dim, H, W), channel
             ordering ``[marker c][field k_in][component r]`` — each BL-regular
             field is a contiguous block of ``repr_dim`` channels. Internal
             layout follows ESCNN's ``bl_regular_representation`` (introspected
             at construction); the "full" basis does not assume a fixed layout.
    Output : ``GeometricTensor`` of type ``[bl_repr] * output_fields``, tensor
             shape (B, output_fields * repr_dim, H, W).

    Bias (optional) is added only to the freq-0 (trivial) component of each
    output field — the only subspace on which a non-zero bias is invariant.
    """

    def __init__(
        self,
        num_channels: int,
        input_fields: int,
        output_fields: int,
        gspace,
        bl_repr,
        use_bias: bool = True,
        intertwiner_basis: str = "scalar",
        module_type: str = "encoder",
    ):
        """
        module_type:
          'encoder'  (default) — input is (B, C·input_fields·R, H, W) where
              C is the number of markers in the batch sample. Aggregates over
              markers: each output field is a marker-conditioned linear
              combination of every input field of every marker. Output:
              (B, output_fields·R, H, W) GeometricTensor.
          'decoder' — input is (B, input_fields·R, H, W) (the BL-regular latent
              from the encoder). Broadcasts to each requested marker, producing
              (B, C, output_fields·R, H, W) GeometricTensor. Used to expand the
              encoder latent into per-marker BL-regular features before the
              equivariant decoder stages.
        """
        super().__init__()
        if module_type not in ("encoder", "decoder"):
            raise ValueError(f"module_type must be 'encoder' or 'decoder', got {module_type!r}")
        self.module_type = module_type
        self.num_channels = num_channels
        self.input_fields = input_fields
        self.output_fields = output_fields
        self.repr_dim = bl_repr.size
        self.use_bias = use_bias
        self.intertwiner_basis = intertwiner_basis

        self.out_type = e2nn.FieldType(gspace, [bl_repr] * output_fields)

        if intertwiner_basis == "scalar":
            # One scalar per (input field, output field) pair — uniform scaling
            self.params_per_pair = 1
            self._irrep_groups = None
        elif intertwiner_basis == "full":
            # Introspect bl_repr to group consecutive identical irreps.
            # For each irrep type with multiplicity m and dim d in the field,
            # the equivariant intertwiner has m_in × m_out × End(irrep) real
            # parameters. End(irrep) = R for finite groups and O(2) (the case
            # used here via flipRot2dOnR2). The action mixes multiplicity copies
            # while leaving the irrep_dim axis untouched.
            G = gspace.fibergroup
            groups = []   # list of (offset, mult, irrep_dim)
            i = 0
            offset = 0
            while i < len(bl_repr.irreps):
                irr_id = bl_repr.irreps[i]
                irrep_dim = G.irrep(*irr_id).size
                mult = 1
                while (
                    i + mult < len(bl_repr.irreps)
                    and bl_repr.irreps[i + mult] == irr_id
                ):
                    mult += 1
                groups.append((offset, mult, irrep_dim))
                offset += mult * irrep_dim
                i += mult
            assert offset == self.repr_dim, (
                f"Irrep grouping consumed {offset} channels, expected {self.repr_dim}"
            )
            self._irrep_groups = groups
            self.params_per_pair = sum(m * m for (_, m, _) in groups)
        else:
            raise ValueError(
                f"intertwiner_basis must be 'scalar' or 'full', got {intertwiner_basis!r}"
            )

        self.weights = nn.Embedding(
            num_channels,
            input_fields * output_fields * self.params_per_pair,
        )
        nn.init.normal_(self.weights.weight, std=input_fields ** -0.5)

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(output_fields))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, indices: torch.Tensor):
        if self.module_type == "encoder":
            return self._forward_encoder(x, indices)
        return self._forward_decoder(x, indices)

    # ------------------------------------------------------------------
    # encoder mode: aggregate over C markers
    # ------------------------------------------------------------------
    def _forward_encoder(self, x: torch.Tensor, indices: torch.Tensor) -> e2nn.GeometricTensor:
        B, total_c, H, W = x.shape
        C = indices.shape[1]
        K_in = self.input_fields
        K_out = self.output_fields
        R = self.repr_dim
        P = self.params_per_pair

        assert total_c == C * K_in * R, (
            f"EquivariantHyperkernel(encoder) expected {C * K_in * R} channels "
            f"(C={C}, K_in={K_in}, R={R}), got {total_c}"
        )

        # (B, C*K_in, R, H, W) — each input field as an atomic repr_dim-vector
        x_f = x.view(B, C * K_in, R, H, W)
        F = C * K_in

        w = self.weights(indices).to(x.dtype)             # (B, C, K_in*K_out*P)
        w = w.view(B, C, K_in, K_out, P)
        w = w.permute(0, 3, 1, 2, 4).reshape(B, K_out, F, P)  # (B, E, F, P)

        out = self._apply_intertwiner_aggregate(x_f, w, B, K_out, R, H, W)

        if self.use_bias:
            add = out.new_zeros(1, K_out, R, 1, 1)
            add[0, :, 0, 0, 0] = self.bias.to(out.dtype)
            out = out + add

        out = out.reshape(B, K_out * R, H, W)
        return e2nn.GeometricTensor(out, self.out_type)

    # ------------------------------------------------------------------
    # decoder mode: broadcast to C markers, per-marker output
    # ------------------------------------------------------------------
    def _forward_decoder(self, x: torch.Tensor, indices: torch.Tensor) -> e2nn.GeometricTensor:
        """
        x:        (B, K_in*R, H, W) — BL-regular latent from the encoder
        indices:  (B, C)            — marker tokens to reconstruct
        returns:  GeometricTensor over the *expanded* field type
                  [bl_repr] * (output_fields * C-effective), tensor shape
                  (B, C, K_out*R, H, W) — but flattened along C·K_out so it
                  can be reshaped by the caller and wrapped per-marker.

        Practically we return the raw tensor as (B, C, K_out*R, H, W) (no
        GeometricTensor) because each sample carries C different marker
        outputs and the caller usually reshapes (B*C, K_out*R, H, W) before
        wrapping as a GeometricTensor of type `[bl_repr]*K_out` for the
        equivariant decoder stages.
        """
        B, total_c, H, W = x.shape
        C = indices.shape[1]
        K_in = self.input_fields
        K_out = self.output_fields
        R = self.repr_dim
        P = self.params_per_pair

        assert total_c == K_in * R, (
            f"EquivariantHyperkernel(decoder) expected {K_in * R} channels "
            f"(K_in={K_in}, R={R}), got {total_c}"
        )

        x_f = x.view(B, K_in, R, H, W)                      # (B, K_in, R, H, W)
        # Weights: (B, C, K_in, K_out, P)
        w = self.weights(indices).to(x.dtype)
        w = w.view(B, C, K_in, K_out, P)

        out = self._apply_intertwiner_broadcast(x_f, w, B, C, K_out, R, H, W)

        if self.use_bias:
            add = out.new_zeros(1, 1, K_out, R, 1, 1)
            add[0, 0, :, 0, 0, 0] = self.bias.to(out.dtype)
            out = out + add

        # (B, C, K_out, R, H, W) → (B, C, K_out*R, H, W)
        return out.reshape(B, C, K_out * R, H, W)

    # ------------------------------------------------------------------
    # shared intertwiner kernels (scalar / full)
    # ------------------------------------------------------------------
    def _apply_intertwiner_aggregate(self, x_f, w, B, K_out, R, H, W):
        """encoder-mode contraction: sum over F = C*K_in input fields → (B, K_out, R, H, W)."""
        if self.intertwiner_basis == "scalar":
            return torch.einsum("bfrhw,bef->berhw", x_f, w[..., 0])
        F = x_f.shape[1]
        out = x_f.new_empty(B, K_out, R, H, W)
        param_offset = 0
        for (offset, mult, irrep_dim) in self._irrep_groups:
            length = mult * irrep_dim
            x_grp = x_f[:, :, offset:offset + length].view(B, F, mult, irrep_dim, H, W)
            w_grp = w[:, :, :, param_offset:param_offset + mult * mult].view(
                B, K_out, F, mult, mult,
            )
            out_grp = torch.einsum("befnm,bfmdhw->bendhw", w_grp, x_grp)
            out[:, :, offset:offset + length] = out_grp.reshape(B, K_out, length, H, W)
            param_offset += mult * mult
        return out

    def _apply_intertwiner_broadcast(self, x_f, w, B, C, K_out, R, H, W):
        """decoder-mode contraction: broadcast over C markers, sum only over K_in.

        x_f shape: (B, K_in, R, H, W)
        w   shape: (B, C, K_in, K_out, P)
        returns:   (B, C, K_out, R, H, W)
        """
        if self.intertwiner_basis == "scalar":
            # out[b,c,e,r,h,w] = Σ_{k_in} w[b,c,k_in,e,0] * x_f[b,k_in,r,h,w]
            return torch.einsum("bkrhw,bcke->bcerhw", x_f, w[..., 0])

        out = x_f.new_empty(B, C, K_out, R, H, W)
        param_offset = 0
        for (offset, mult, irrep_dim) in self._irrep_groups:
            length = mult * irrep_dim
            x_grp = x_f[:, :, offset:offset + length].view(
                B, self.input_fields, mult, irrep_dim, H, W,
            )
            w_grp = w[:, :, :, :, param_offset:param_offset + mult * mult].view(
                B, C, self.input_fields, K_out, mult, mult,
            )
            # out_grp[b,c,e,n,d,h,w] = Σ_{k_in, m_in} w_grp[b,c,k_in,e,n,m_in] * x_grp[b,k_in,m_in,d,h,w]
            out_grp = torch.einsum("bckenm,bkmdhw->bcendhw", w_grp, x_grp)
            out[:, :, :, offset:offset + length] = out_grp.reshape(
                B, C, K_out, length, H, W,
            )
            param_offset += mult * mult
        return out


class EquivariantMultiplexImageEncoder(nn.Module):
    """Encoder backbone for encoding multiplex images.

    v2 differences vs. v1:
      * MA encoder does *not* project to trivial at its exit — the entire
        MA → Hyperkernel → PM path stays in BL-regular GeometricTensors.
      * ``EquivariantHyperkernel`` is a field-wise marker-conditioned mixer
        (scalar weights per (input field, output field) pair), keeping the
        pipeline continuously equivariant end-to-end.
      * Strided downsampling uses antialiased Gaussian blur + 1×1 channel
        expander, addressing the main source of equivariance error under
        non-discrete rotations.
    """

    def __init__(
            self,
            num_channels: int,
            ma_layers_blocks,
            ma_embedding_dims,
            hyperkernel_config,
            pm_layers_blocks,
            pm_embedding_dims,
            maximum_frequency,
            include_stem,
            latent_nonlinearity,
            ma_maximum_frequency: Optional[int] = None,
            use_gating: bool = True,
            use_norm: bool = False,
            use_layerscale: bool = True,
            layerscale_init: float = 1e-6,
            gate_bias_init: float = 1.0,
            pool_use_act: bool = True,
            antialiased_downsample: bool = True,
            antialiased_stem: Optional[bool] = None,
            output_scalars: int = None,
            output_irreps=None,
            output_trivial: bool = True,
    ):
        """Initialize the Multiplex Image Encoder.

        Args:
            num_channels (int): Number of all possible channels/markers.
            ma_layers_blocks (List[int]): Number of blocks in each marker-agnostic layer.
            ma_embedding_dims (List[int]): Embedding dimensions for each marker-agnostic layer.
            hyperkernel_config (Dict): Configuration for the hyperkernel.
            pm_layers_blocks (List[int]): Number of blocks in each pan-marker layer.
            pm_embedding_dims (List[int]): Embedding dimensions for each pan-marker layer.
        """
        super().__init__()

        # When ma_maximum_frequency is unset, MA shares max_freq with PM
        # (backward-compatible behaviour). Setting it to a smaller value lets
        # the MA encoder run cheaper per-marker feature extraction without
        # losing PM's angular expressivity downstream — e.g. ma_max_freq=1
        # uses 6 channels per BL-regular field vs M=2's 10, ~40% cheaper at
        # the spatially largest stage.
        ma_max_freq = (
            maximum_frequency if ma_maximum_frequency is None
            else int(ma_maximum_frequency)
        )
        self._ma_max_freq = ma_max_freq

        # Single shared gspace so field types built here are compatible with
        # field types built inside each encoder (escnn identifies modules by
        # object identity of the gspace, not by equality).
        # NOTE: gspace's maximum_frequency caps the irrep cache. R2Conv needs to
        # build kernel bases via Clebsch-Gordan tensor products of all pairs of
        # irreps in the BL-regular field type, which can produce irreps up to
        # 2*max_freq. Use 2 × max(ma, pm) so both encoders' bases fit.
        gspace_cap = 2 * max(ma_max_freq, maximum_frequency)
        self._gspace = escnn.gspaces.flipRot2dOnR2(
            N=-1, maximum_frequency=gspace_cap
        )

        # channel-agnostic part — returns a BL-regular GeometricTensor at
        # ma_max_freq (NOT necessarily the PM frequency).
        self.marker_agnostic_encoder = EquivariantConvNeXtEncoder(
            input_channels=1,
            layers_blocks=ma_layers_blocks,
            embedding_dims=ma_embedding_dims,
            maximum_frequency=ma_max_freq,
            latent_nonlinearity=latent_nonlinearity,
            use_gating=use_gating,
            use_norm=use_norm,
            use_layerscale=use_layerscale,
            layerscale_init=layerscale_init,
            gate_bias_init=gate_bias_init,
            pool_use_act=pool_use_act,
            antialiased_downsample=antialiased_downsample,
            antialiased_stem=antialiased_stem,
            output_trivial=False,           # keep BL-regular through MA→PM
            gspace=self._gspace,
        )

        # The equivariant hyperkernel is pointwise (k=1) by construction.
        for k in ("kernel_size", "stride"):
            if k in hyperkernel_config and hyperkernel_config[k] != 1:
                raise ValueError(
                    f"EquivariantHyperkernel is pointwise; got {k}={hyperkernel_config[k]}"
                )
        if hyperkernel_config.get("padding", 0) not in (0, None):
            raise ValueError(
                f"EquivariantHyperkernel is pointwise; got padding={hyperkernel_config['padding']}"
            )

        # PM-side BL-regular: the field type the hyperkernel and PM encoder
        # operate in. May differ from MA's bl_repr when ma_max_freq < maximum_frequency.
        bl_repr_pm = self._gspace.fibergroup.bl_regular_representation(
            maximum_frequency
        )
        self._bl_repr_pm = bl_repr_pm

        K_out = hyperkernel_config["embedding_dim"]
        self.hyperkernel = EquivariantHyperkernel(
            num_channels=num_channels,
            input_fields=ma_embedding_dims[-1],
            output_fields=K_out,
            gspace=self._gspace,
            bl_repr=bl_repr_pm,
            use_bias=hyperkernel_config.get("use_bias", True),
            intertwiner_basis=hyperkernel_config.get("intertwiner_basis", "scalar"),
        )

        # MA → hyperkernel adapter. When ma_max_freq == maximum_frequency the
        # bl_reprs match and no adapter is needed. Otherwise, an equivariant
        # 1×1 R2Conv lifts the MA output type ([bl_repr_ma] * ma_dim) into the
        # PM type ([bl_repr_pm] * ma_dim). By Schur's lemma it can only
        # populate the irrep components shared between the two BL-regular
        # representations — i.e. trivial + sign + psi_k for k ≤ ma_max_freq.
        # The higher-frequency components of the PM type start at zero (or
        # bias on the trivial component) and are generated by the subsequent
        # field-wise hyperkernel + PM blocks.
        if ma_max_freq != maximum_frequency:
            hk_input_type = e2nn.FieldType(
                self._gspace, [bl_repr_pm] * ma_embedding_dims[-1]
            )
            self.ma_to_hk_adapter = e2nn.R2Conv(
                self.marker_agnostic_encoder.out_type,
                hk_input_type,
                kernel_size=1, stride=1, padding=0,
                bias=True, initialize=True,
            )
        else:
            self.ma_to_hk_adapter = None

        # Equivariant nonlinearity on the BL-regular field type
        self.act = e2nn.NormNonLinearity(self.hyperkernel.out_type)

        # pan-marker part — receives the hyperkernel's BL-regular field type
        self.pan_marker_encoder = EquivariantConvNeXtEncoder(
            input_channels=K_out,
            layers_blocks=pm_layers_blocks,
            embedding_dims=pm_embedding_dims,
            include_stem=False,
            maximum_frequency=maximum_frequency,
            latent_nonlinearity=latent_nonlinearity,
            use_gating=use_gating,
            use_norm=use_norm,
            use_layerscale=use_layerscale,
            layerscale_init=layerscale_init,
            gate_bias_init=gate_bias_init,
            pool_use_act=pool_use_act,
            antialiased_downsample=antialiased_downsample,
            # output_trivial: True  → final latent as plain tensor of trivial
            #                         scalars (the original/default modelv2 path).
            # output_trivial: False → return a BL-regular GeometricTensor instead
            #                         (used by FullyEquivariantMultiplexAutoencoderV2
            #                         so the BL-regular latent flows straight into
            #                         the equivariant decoder — no Regular2Trivial /
            #                         trivial2regular round-trip at the bottleneck).
            output_trivial=output_trivial,
            output_scalars=output_scalars,  # only consulted if output_trivial=True
            # Mixed-irrep latent (Option B). When set, overrides output_scalars
            # and emits ``[trivial, sign, psi_k, ...]`` instead of trivial-only.
            output_irreps=output_irreps,
            input_field_type=self.hyperkernel.out_type,
            gspace=self._gspace,
        )

    def forward(self, x: torch.Tensor, encoded_indices: torch.Tensor, return_features: bool = False) -> Dict:
        """Forward pass of the encoder.

        Args:
            x (torch.Tensor): Multiplex images batch tensor with shape [B, C, H, W]
            encoded_indices (torch.Tensor): Indices of the markers in channels tensor with shape [B, C].
            return_features (bool, optional): If True, returns the features after each block. Defaults to False.

        Returns:
            Dict: A dictionary containing the output tensor and optionally the features.
        """
        outputs = {}
        features = []

        B, C, H, W = x.shape
        x = x.reshape(B * C, 1, H, W)
        x = self.marker_agnostic_encoder(x, return_features=return_features)
        if return_features:
            features += x['features']
        x = x['output']  # GeometricTensor of shape (B*C, K_in*R_ma, H_ma, W_ma)

        # Lift MA's bl_repr_ma fields into the PM bl_repr if they differ.
        # After this the per-field repr_dim is the PM one — required by the
        # hyperkernel's contiguous-block reshape below.
        if self.ma_to_hk_adapter is not None:
            x = self.ma_to_hk_adapter(x)

        t = x.tensor
        _, CR, H_ma, W_ma = t.shape  # CR = K_in * R_pm (fields×repr contiguous)
        # Collapse the marker axis into the field axis; each BL-regular field
        # stays a contiguous repr_dim block — required by EquivariantHyperkernel.
        t = t.view(B, C, CR, H_ma, W_ma).reshape(B, C * CR, H_ma, W_ma)

        x = self.hyperkernel(t, encoded_indices)   # GeometricTensor [B, K_out*R_pm, H_ma, W_ma]
        x = self.act(x)

        x = self.pan_marker_encoder(x, return_features=return_features)
        if return_features:
            features += x['features']
        x = x['output']

        outputs['output'] = x
        if return_features:
            outputs['features'] = features

        # print(f"Latent shape: {x.shape}")
        return outputs


class EquivariantMultiplexAutoencoder(nn.Module):
    """Multiplex image Autoencoder with Superkernel and Multiplex Image Encoder-Decoder."""

    def __init__(
            self,
            num_channels: int,
            encoder_config: Dict,
            decoder_config: Dict,
            ):
        """Initialize the Multiplex Autoencoder model.

        Args:
            num_channels (int): Number of all possible channels/markers.
            superkernel_config (Dict): Configuration for the superkernel.
            encoder_config (Dict): Configuration for the encoder.
            decoder_config (Dict): Configuration for the decoder.
        """
        super().__init__()
        # Determine the latent channel count the decoder must consume.
        # Precedence:
        #   1. ``output_irreps``  → mixed-irrep latent, total = Σ mult·irrep_dim
        #                          (Option B from the encoder design discussion;
        #                          uses trivial + sign [+ psi_k] per input field
        #                          instead of just the trivial sub-component).
        #   2. ``output_scalars`` → trivial-only Regular2Trivial; this is the
        #                          actual latent channel count.
        #   3. fallback           → ``pm_embedding_dims[-1]`` (= number of
        #                          regular fields at the last PM stage; matches
        #                          Regular2Trivial's default with rank-equal
        #                          trivial output).
        output_irreps = encoder_config.get('output_irreps')
        if output_irreps is not None:
            self.latent_dim = mixed_irreps_total_channels(output_irreps)
        else:
            self.latent_dim = encoder_config.get(
                'output_scalars', None
            ) or encoder_config['pm_embedding_dims'][-1]
        # self.decoder_dim = decoder_config['decoded_embed_dim']
        self.num_channels = num_channels
  
        self.act = nn.GELU()

        # self.pixel_shift_superkernel = Superkernel(
        #     num_channels=self.num_channels,
        #     embedding_dim=1,
        #     layer_type='linear',
        #     num_layers=0,
        #     num_heads=None,
        #     mlp_ratio=None,
        #     kernel_size=None,
        # )

        self.encoder = EquivariantMultiplexImageEncoder( # finish thi
            num_channels=self.num_channels,
            # num_all_channels=self.num_channels,
            **encoder_config
        )

        hyperkernels_scaling_factor = (
            encoder_config['hyperkernel_config']['stride']
            * decoder_config['hyperkernel_config']['stride']
        )
        scaling_factor = hyperkernels_scaling_factor * 2 ** len(
            encoder_config['ma_layers_blocks'] + encoder_config['pm_layers_blocks'][:-1]
        )
        self.decoder = MultiplexImageDecoder(
            input_embedding_dim=self.latent_dim,
            scaling_factor=scaling_factor,
            num_channels=self.num_channels,
            **decoder_config
        )


    def encode_images(
            self, 
            x: torch.Tensor, 
            encoded_indices: torch.Tensor,
            return_features: bool = False,
        ) -> Dict:
        """Encode the input images using the encoder.

        Args:
            x (torch.Tensor): Input images tensor with shape (B, C, H, W).
            encoded_indices (torch.Tensor): Indices of the markers in channels.
            return_features (bool, optional): If True, returns the features after encoding. Defaults to False.

        Returns:
            Dict: A dictionary containing the encoded images tensor (under 'output') and optionally the features.
        """
        encoding_output = self.encoder(x, encoded_indices=encoded_indices, return_features=return_features)
        outputs = {'output': encoding_output['output']}
        # print(f'Encoding output shape: {outputs["output"].shape}')

        if return_features:
            outputs['features'] = encoding_output['features']
        return outputs

    def decode_images(
            self, 
            x: torch.Tensor, 
            decoded_indices: torch.Tensor,
        ) -> torch.Tensor:
        """Decode the encoded images using the decoder.

        Args:
            x (torch.Tensor): Encoded images tensor with shape (B, E', H', W').
            decoded_indices (torch.Tensor): Indices of the markers in channels for decoding.

        Returns:
            torch.Tensor: Decoded images tensor with shape (B, C, H, W).
        """
        x = self.decoder(x, decoded_indices)
        return x

    def forward(
            self, 
            x: torch.Tensor, 
            encoded_indices: torch.Tensor, 
            decoded_indices: torch.Tensor,
            return_features: bool = False,
        ) -> Dict:
        """Forward pass of the Multiplex Autoencoder.

        Args:
            x (torch.Tensor): Input images tensor with shape (B, C, H, W).
            encoded_indices (torch.Tensor): Indices of the markers in channels
                for encoding.
            decoded_indices (torch.Tensor): Indices of the markers in channels
                for decoding.

        Returns:
            Dict: A dictionary containing the reconstructed images tensor (under 'output') and optionally the features.
        """
        encoding_output = self.encode_images(x, encoded_indices, return_features=return_features)
        x = encoding_output['output']
        x = self.decode_images(x, decoded_indices)
        outputs = {'output': x}
        if return_features:
            outputs['features'] = encoding_output['features']
        return outputs




class EscnnMultiplexImageEncoder(nn.Module):
    """Encoder backbone for encoding multiplex images."""

    def __init__(
            self,
            encoder_class: Type,
            reshape_fn: Callable = nn.Identity(),
            **kwargs
    ):
        """Initialize the Multiplex Image Encoder.

        Args:
            encoder_class (Type): Encoder class to use.
            reshape_fn (Type, optional): Reshape function to apply to the output of the encoder. Defaults to nn.Identity.
        """
        super().__init__()
        self.encoder = encoder_class(**kwargs)
        self.last_conv = Regular2Trivial(self.encoder.out_type, n_scalars=kwargs["embedding_dims"][-1])
        self.reshape_fn = reshape_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.encoder.forward_features(x)
        x = self.last_conv(x)  # [B, C, H, W] → [B, E, H, W]
        x = x.tensor  # Extract the raw tensor from GeometricTensor
        x = self.reshape_fn(x)
        return x



class Regular2Trivial(nn.Module):
    """
    Point-wise R2Conv that projects any FieldType to *n_scalars*
    copies of the trivial representation.
    """

    def __init__(self, in_type: e2nn.FieldType, n_scalars: int = 1):
        super().__init__()

        gspace   = in_type.gspace
        out_type = e2nn.FieldType(gspace,
                                  [gspace.trivial_repr] * n_scalars)

        # kernel_size = 1 → no spatial mixing, only channel mixing
        self.proj = e2nn.R2Conv(in_type, out_type,
                                kernel_size=1, bias=True,
                                initialize=True # TODO initialize=True
                                )

        self.in_type, self.out_type = in_type, out_type

    def forward(self, x: e2nn.GeometricTensor):
        return self.proj(x)


# ---------------------------------------------------------------------------
# Mixed-irrep projection for Option B from the encoder design discussion.
# Generalises Regular2Trivial: instead of projecting to only the trivial
# sub-component of each input field (which throws away repr_dim - 1 channels
# per field by Schur's lemma), this projects to a configurable mix of irreps
# — trivial, sign, and any psi_k up to the encoder's max_freq. The latent uses
# more of the regular-field information while staying equivariant; the
# downstream non-equivariant decoder consumes the result as plain channels.
# ---------------------------------------------------------------------------
_IRREP_DIM_BY_NAME = {'trivial': 1, 'sign': 1}


def _irrep_dim(name: str) -> int:
    if name in _IRREP_DIM_BY_NAME:
        return _IRREP_DIM_BY_NAME[name]
    if name.startswith('psi_'):
        return 2
    raise ValueError(f"Unknown irrep name: {name!r}")


def _irrep_name_to_id(name: str):
    """Map friendly irrep names to escnn O(2) irrep ids ``(j, k)``."""
    if name == 'trivial':
        return (0, 0)
    if name == 'sign':
        return (1, 0)
    if name.startswith('psi_'):
        k = int(name.split('_', 1)[1])
        if k < 1:
            raise ValueError(f"psi_{k} requires k >= 1")
        return (1, k)
    raise ValueError(f"Unknown irrep name: {name!r}")


def _normalise_irrep_mix(irrep_mix):
    """Accept a list of ``[name, mult]`` pairs (or tuples), drop mult<=0."""
    norm = []
    for entry in irrep_mix:
        if isinstance(entry, dict):
            # also accept {'name': 'trivial', 'mult': 384}
            name, mult = entry['name'], entry['mult']
        else:
            name, mult = entry[0], int(entry[1])
        if mult > 0:
            norm.append((name, int(mult)))
    return norm


def mixed_irreps_total_channels(irrep_mix) -> int:
    """Total channel count of a mixed-irrep latent: ``Σ mult_i · dim_i``."""
    return sum(mult * _irrep_dim(name) for name, mult in _normalise_irrep_mix(irrep_mix))


class RegularToMixedIrreps(nn.Module):
    """Point-wise R2Conv from a BL-regular FieldType to a configurable mix
    of irreps.

    By Schur's lemma, an equivariant map from regular_1 to trivial only uses
    the trivial sub-component (1-dim) of each input field — discarding sign
    and any psi_k components. By emitting a richer mix, this projection lets
    the latent carry information from those discarded sub-components, while
    remaining equivariant: under rotation/flip the trivial channels stay
    invariant, sign channels flip on reflection, and psi_k channels transform
    as a steerable 2-D vector at frequency k.

    The downstream non-equivariant decoder treats the resulting tensor as
    plain channels (it cannot exploit the rotational structure of sign/psi_k
    components), but the encoder is no longer bandwidth-bottlenecked at this
    last 1×1 layer.

    Args:
        in_type:    input FieldType, typically ``[bl_repr] * N``.
        irrep_mix:  list of ``(name, multiplicity)`` pairs. Recognised names:
                    ``'trivial'`` (1-dim), ``'sign'`` (1-dim),
                    ``'psi_k'`` for k=1..max_freq (2-dim each).
                    Total output channels = ``Σ mult_i · dim_i``.

    Example (M=2, 768 output channels):
        irrep_mix = [('trivial', 192), ('sign', 192),
                     ('psi_1', 96), ('psi_2', 96)]
        → 192 + 192 + 192 + 192 = 768 channels, using all four irrep types.
    """

    def __init__(self, in_type: e2nn.FieldType, irrep_mix):
        super().__init__()

        gspace = in_type.gspace
        G      = gspace.fibergroup
        spec   = _normalise_irrep_mix(irrep_mix)
        if not spec:
            raise ValueError("irrep_mix must contain at least one entry with mult > 0")

        reprs = []
        for name, mult in spec:
            irrep = G.irrep(*_irrep_name_to_id(name))
            reprs.extend([irrep] * mult)
        out_type = e2nn.FieldType(gspace, reprs)

        # kernel_size = 1 → channel mix only, no spatial mixing
        self.proj = e2nn.R2Conv(
            in_type, out_type,
            kernel_size=1, bias=True, initialize=True,
        )

        self.in_type, self.out_type = in_type, out_type
        self.irrep_mix = spec
        self.total_channels = out_type.size

    def forward(self, x: e2nn.GeometricTensor):
        return self.proj(x)

def apply_bank_gate(mid_feat: e2nn.GeometricTensor, gate: torch.Tensor, repr_dim: int) -> e2nn.GeometricTensor:
    """
    mid_feat: GeometricTensor with FieldType = [bl_repr] * (exp*K)
             tensor shape [B, (exp*K*repr_dim), H, W]
    gate:    raw tensor shape [B, exp*K, H, W] (trivial scalars)
    repr_dim: bl_repr.size

    returns: GeometricTensor same type as mid_feat
    """
    ft = mid_feat.tensor
    B, C, H, W = ft.shape
    assert C % repr_dim == 0
    n_banks = C // repr_dim  # exp*K

    g = torch.sigmoid(gate).view(B, n_banks, 1, H, W)  # [B, n_banks, 1, H, W]
    ft = ft.view(B, n_banks, repr_dim, H, W) * g
    ft = ft.view(B, n_banks * repr_dim, H, W)

    return e2nn.GeometricTensor(ft, mid_feat.type)

class FieldLayerScale(nn.Module):
    """
    Equivariant LayerScale: one scalar per *field copy* (uniform across irrep components).
    """
    def __init__(self, ftype: e2nn.FieldType, init_value: float = 1e-6):
        super().__init__()
        self.ftype = ftype

        # build field_indices: for each channel, which field it belongs to
        start = 0
        fields = []
        for r in ftype:
            size = r.size
            fields.append((slice(start, start+size), size))
            start += size

        field_indices = torch.empty(start, dtype=torch.long)
        for idx, (sl, _) in enumerate(fields):
            field_indices[sl] = idx
        self.register_buffer("field_indices", field_indices)
        self.n_fields = len(fields)

        # gamma per field
        self.gamma = nn.Parameter(init_value * torch.ones(1, self.n_fields, 1, 1))

    def forward(self, x: e2nn.GeometricTensor) -> e2nn.GeometricTensor:
        t = x.tensor  # [B, C, H, W]
        B, C, H, W = t.shape

        idx = self.field_indices.view(1, C, 1, 1).expand(B, -1, -1, -1)
        gamma = self.gamma.expand(B, -1, -1, -1)
        gamma_per_channel = torch.gather(gamma, 1, idx)

        return e2nn.GeometricTensor(t * gamma_per_channel, x.type)


class BLConvNeXtBlock(nn.Module):
    def __init__(self,
                 in_type: e2nn.FieldType,
                 expansion: int = 4,
                 ksize: int = 7,
                 use_grn: bool = False,
        use_gating: bool = True,
        use_norm: bool = False,
        use_layerscale: bool = True,
        layerscale_init: float = 1e-6,
        gate_bias_init: float = 1.0,   # >0 starts gates ~open
    ):
        super().__init__()

        self.use_grn = use_grn
        self.use_gating = use_gating
        self.use_norm = use_norm
        self.use_layerscale = use_layerscale

        # initialize=False
        initialize=True

        gspace = in_type.gspace
        K = len(in_type)   
        repr_dim = in_type.representations[0].size
        self._repr_dim = repr_dim

        # For the point-wise MLP we keep the same irrep set but multiply the copies
        mid_type = e2nn.FieldType(
            gspace,
            expansion * list(in_type.representations)
        )

        # depth-wise ⇒ no mixing across *copies*
        # easiest robust choice: a *full* equivariant conv
        self.depthwise = e2nn.R2Conv(
            in_type, in_type,
            kernel_size=ksize, padding=ksize//2, bias=True,
            # groups=len(in_type.representations) // bank_length,
            groups=K,
            initialize=initialize
        )


        # WARNING norm modification
        if use_norm:
            self.norm = EquivariantPixelLN(in_type, eps=1e-6, center_scalar=True, affine=True)
        
        mid_feat_type = e2nn.FieldType(gspace, list(in_type.representations) * expansion)
        self.mid_feat_type = mid_feat_type

        assert len({r.size for r in in_type.representations}) == 1
        assert len({r.size for r in self.mid_feat_type.representations}) == 1


        self.pw_up_feat = e2nn.R2Conv(in_type, mid_feat_type, kernel_size=1, bias=True, initialize=True)
        self.act = e2nn.NormNonLinearity(mid_feat_type)

        # 4) gates: one trivial scalar per expanded bank
        if use_gating:
            gate_type = e2nn.FieldType(gspace, [gspace.trivial_repr] * (expansion * K))
            self.pw_up_gate = e2nn.R2Conv(in_type, gate_type, kernel_size=1, bias=True, initialize=True)

            # initialize gate bias so sigmoid ~ open at start
            if getattr(self.pw_up_gate, "bias", None) is not None and self.pw_up_gate.bias is not None:
                with torch.no_grad():
                    self.pw_up_gate.bias.fill_(gate_bias_init)

        # 5) optional GRN
        if use_grn:
            self.grn = GRNByIrrep(mid_feat_type)

        # 6) project back
        self.pw_down = e2nn.R2Conv(mid_feat_type, in_type, kernel_size=1, bias=True, initialize=True)

        # 7) optional LayerScale
        if use_layerscale:
            self.ls = FieldLayerScale(in_type, init_value=layerscale_init)
        self.in_type  = in_type
        self.out_type = in_type

    def forward(self, x: e2nn.GeometricTensor):
        y = self.depthwise(x)
        if self.use_norm:
            y = self.norm(y)

        feat = self.pw_up_feat(y)
        feat = self.act(feat)

        if self.use_grn:
            feat = self.grn(feat)

        if self.use_gating:
            gate = self.pw_up_gate(y).tensor  # [B, exp*K, H, W]
            feat = apply_bank_gate(feat, gate, repr_dim=self._repr_dim)

        y = self.pw_down(feat)

        if self.use_layerscale:
            y = self.ls(y)

        return x + y

# ---------------------------------------------------------------------
# 1.  A stack of BL-ConvNeXt blocks that preserves the FieldType
# ---------------------------------------------------------------------
class EscnnConvNeXtBlocks(nn.Module):
    """
    Sequence of `num_blocks` BLConvNeXtBlock, all sharing the *same* FieldType.
    """

    def __init__(
        self,
        in_type: e2nn.FieldType,
        num_blocks: int = 1,
        ksize: int = 7,
        expansion: int = 4,
        use_grn: bool = False,
        use_gating: bool = True,
        use_norm: bool = False,
        use_layerscale: bool = True,
        layerscale_init: float = 1e-6,
        gate_bias_init: float = 1.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                BLConvNeXtBlock(
                    in_type=in_type,
                    ksize=ksize,
                    expansion=expansion,
                    use_grn=use_grn,
                    use_gating=use_gating,
                    use_norm=use_norm,
                    use_layerscale=use_layerscale,
                    layerscale_init=layerscale_init,
                    gate_bias_init=gate_bias_init,
                )
                for _ in range(num_blocks)
            ]
        )
        self.in_type = self.out_type = in_type

    def forward(self, x: e2nn.GeometricTensor):
        for blk in self.blocks:
            x = blk(x)
        return x
    

class EquivariantConvNeXtEncoder(nn.Module):
    r"""
    *Input*  : B × C_in × H × W  (plain tensor, **C_in = `channel_embedding_dim`**)  
    *Output* : GeometricTensor whose FieldType is the last stage’s one.
    """

    def __init__(
        self,
        input_channels,
        layers_blocks,            # e.g. [2, 2, 2]
        embedding_dims,           # e.g. [192, 384, 768]  ← **must** be multiples of (1+2*max_freq)
        # channel_embedding_dim,    # e.g. 96 trivial copies
        include_stem: bool = True,
        maximum_frequency: int = 3,
        use_grn: bool = False,
        use_gating: bool = True,
        use_norm: bool = False,
        use_layerscale: bool = True,
        layerscale_init: float = 1e-6,
        gate_bias_init: float = 1.0,
        latent_nonlinearity="none",
        pool_use_act: bool = True,
        antialiased_downsample: bool = True,
        antialiased_stem: Optional[bool] = None,
        output_trivial: bool = True,
        output_scalars: int = None,
        output_irreps=None,
        input_field_type=None,
        gspace=None,
    ):
        super().__init__()
        self.latent_nonlinearity = latent_nonlinearity
        self.pool_use_act = pool_use_act
        self.use_norm = use_norm
        self.output_trivial = output_trivial
        self.antialiased_downsample = antialiased_downsample
        # output_irreps takes precedence over output_scalars when both are set.
        # See ``RegularToMixedIrreps`` for the format: list of (name, mult)
        # pairs, e.g. ``[('trivial', 384), ('sign', 384)]`` for an M=1 latent
        # that uses both freq-0 irreps from each input regular field.
        self.output_irreps = output_irreps
        # antialiased_stem=None → inherit from antialiased_downsample (backward compat).
        # Explicit False disables antialiasing only at the stem (stage 0 with
        # include_stem=True). Useful to preserve fine-grained input detail for
        # reconstruction while still antialiasing deeper stages where aliasing
        # under continuous rotations is most harmful.
        self.antialiased_stem = (
            antialiased_downsample if antialiased_stem is None else antialiased_stem
        )
        self.output_scalars = output_scalars

        # ───────────────────────────────────────────────────────────────
        # 0.  Group bookkeeping
        # ───────────────────────────────────────────────────────────────
        self.max_freq  = maximum_frequency
        if gspace is None:
            # self.r2_act    = escnn.gspaces.rot2dOnR2(N=-1, maximum_frequency=2 * maximum_frequency)        # SO(2)
            # NOTE: gspace's maximum_frequency = 2 * max_freq to ensure ESCNN's
            # irrep cache contains the irreps needed for Clebsch-Gordan tensor
            # product decompositions inside R2Conv kernel bases.
            self.r2_act    = escnn.gspaces.flipRot2dOnR2(N=-1, maximum_frequency=2 * maximum_frequency)        # SO(2) - axis zero was throwing
            # self.r2_act    = e2cnn.gspaces.Rot2dOnR2(N=-1, maximum_frequency=2 * maximum_frequency)        # SO(2)
        else:
            self.r2_act = gspace
        self.G        = self.r2_act.fibergroup
        bl_repr        = self.G.bl_regular_representation(self.max_freq)  # dim = 1+2*max_freq
        self.bl_repr = bl_repr
        # irreps = [self.G.irrep(0)] + [self.G.irrep(k) for k in range(1, self.max_freq + 1)]
        # bl_repr = directsum(irreps, name=f"bl_reg_{self.max_freq}")
        # print(bl_repr)
        repr_dim       = 1 + 2 * self.max_freq                           # 7 for max_freq = 3
       
        # ------------------------------------------------------------------
        # 1.  Build FieldTypes for every stage
        # ------------------------------------------------------------------
        def _make_stage_type(channels: int) -> e2nn.FieldType:
            # if channels % repr_dim != 0:
            #     raise ValueError(
            #         f"embedding_dim (= {channels}) must be a multiple of "
            #         f"the BL-regular representation dimension (= {repr_dim})"
            #     )
            # reps = [bl_repr] * (channels // repr_dim)
            reps = [bl_repr] * channels
            # irrep_reps = [self.G.irrep(*irr) for irr in bl_repr.irreps]

            # teraz "channels" oznacza liczbę kopii całego bl-banku (tak jak wcześniej)
            # reps = irrep_reps * channels
            # print(reps)
            return e2nn.FieldType(self.r2_act, reps)

        if input_field_type is not None:
            self.input_type = input_field_type
        else:
            self.input_type = e2nn.FieldType(
                self.r2_act, [self.r2_act.trivial_repr] * input_channels
            )

        stage_types = [_make_stage_type(d) for d in embedding_dims]

        # ------------------------------------------------------------------
        # 2.  Stem / inter-stage down-sampling convolutions
        #     (stride-2, 2×2 kernel, no padding = “pixel unshuffle” like ConvNeXt)
        self.poolings = nn.ModuleList()
        prev_type = self.input_type

        # ------------------------------------------------------------------
        for idx, out_type in enumerate(stage_types):
            if idx == 0 and not include_stem:
                # Channel/type projection only — no spatial downsampling.
                # The hyperkernel outputs trivial irreps; blocks expect regular irreps.
                # A 1×1 R2Conv handles the type conversion without striding.
                # (mirrors original stem=False → nn.Identity() in the non-equivariant case
                # where types always match; here types differ so we need the projection.)
                self.poolings.append(
                    nn.Sequential(
                        e2nn.R2Conv(
                            prev_type, out_type,
                            kernel_size=1, stride=1, padding=0,
                            bias=True, initialize=True,
                        ),
                    )
                )
            else:
                # The "stem" is stage 0 of an include_stem=True encoder (i.e.
                # operates on the original input resolution). Allow it to opt
                # out of antialiasing independently — useful when reconstruction
                # of fine-grained input detail matters and you'd rather not
                # low-pass the input before the first downsampling step.
                is_stem = (idx == 0 and include_stem)
                use_antialiased = self.antialiased_stem if is_stem else antialiased_downsample
                if use_antialiased:
                    # Antialiased pool (Gaussian blur + stride-2) followed by a
                    # 1×1 channel expander. The blur uses default centered padding
                    # p=(k-1)//2 so output is ceil(H/2) — identical shape to the
                    # k=3/s=2/p=1 conv path for typical inputs. See
                    # escnn.nn.PointwiseAvgPoolAntialiased2D.
                    downsample = nn.Sequential(
                        e2nn.PointwiseAvgPoolAntialiased(
                            prev_type, sigma=0.66, stride=2,
                        ),
                        e2nn.R2Conv(
                            prev_type, out_type,
                            kernel_size=1, stride=1, padding=0,
                            bias=True, initialize=True,
                        ),
                    )
                else:
                    downsample = nn.Sequential(
                        e2nn.R2Conv(
                            prev_type, out_type,
                            kernel_size=4, stride=2, padding=1,
                            bias=True, initialize=True,
                        ),
                    )
                self.poolings.append(downsample)
            prev_type = out_type

        # ------------------------------------------------------------------
        # 3.  Stage-wise NormNonLinearities (GELU analogue) and blocks
        # ------------------------------------------------------------------
        # WARNING norm modification
        if use_norm:
            self.pool_norms = nn.ModuleList(
                [EquivariantPixelLN(t, eps=1e-6, center_scalar=True, affine=True) for t in stage_types]
            )

        if self.pool_use_act:
            self.acts = nn.ModuleList(
                [e2nn.NormNonLinearity(t) for t in stage_types]
            )

        self.blocks = nn.ModuleList(
            [
                EscnnConvNeXtBlocks(
                    in_type=t,
                    num_blocks=n,
                    use_grn=use_grn,
                    use_gating=use_gating,
                    use_norm=use_norm,
                    use_layerscale=use_layerscale,
                    layerscale_init=layerscale_init,
                    gate_bias_init=gate_bias_init,
                )
                for t, n in zip(stage_types, layers_blocks)
            ]
        )

        self.out_type = self.blocks[-1].out_type  # last stage’s FieldType

        if self.output_trivial:
            # Variable kept as ``self.regular2trivial`` for backward
            # compatibility with the forward path; can now hold either
            # ``Regular2Trivial`` (trivial-only, default) or
            # ``RegularToMixedIrreps`` (mixed-irrep output, Option B).
            if self.output_irreps is not None:
                self.regular2trivial = RegularToMixedIrreps(
                    in_type=self.out_type,
                    irrep_mix=self.output_irreps,
                )
            else:
                # If output_scalars not specified, default to number of fields (backward compatible)
                n_scalars = self.output_scalars if self.output_scalars is not None else embedding_dims[-1]
                self.regular2trivial = Regular2Trivial(
                    in_type=self.out_type,
                    n_scalars=n_scalars
                )
        else:
            self.regular2trivial = None

    # a convenience wrapper identical to vanilla ConvNeXt API
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


    def _postprocess_latent(self, x: torch.Tensor) -> torch.Tensor:
        if self.latent_nonlinearity in ("asinh", "arcsinh"):
            return torch.asinh(x)
        elif self.latent_nonlinearity == "none":
            return x
        else:
            raise ValueError(f"Unknown latent nonlinearity: {self.latent_nonlinearity}")

    def _forward_impl(self, x, return_features: bool = False) -> Dict:
        outputs = {}
        features = []

        # Accept either a plain tensor (wrapped into the default input type)
        # or a pre-built GeometricTensor (used as-is, with a type check).
        if isinstance(x, e2nn.GeometricTensor):
            assert x.type == self.input_type, (
                "EquivariantConvNeXtEncoder received a GeometricTensor whose "
                "FieldType does not match self.input_type"
            )
            g = x
        else:
            g = e2nn.GeometricTensor(x, self.input_type)

        is_last = lambda i: i == len(self.blocks) - 1
        if self.pool_use_act:
            for i, (pool, act, blk) in enumerate(zip(
                self.poolings, self.acts, self.blocks
            )):
                g = pool(g)
                if self.use_norm:
                    g = self.pool_norms[i](g)
                g = act(g)
                g = blk(g)
                if is_last(i) and self.output_trivial:
                    g = self.regular2trivial(g)
                    g = g.tensor  # Extract the raw tensor from GeometricTensor
                if return_features:
                    features.append(g)
        else:
            for i, (pool, blk) in enumerate(zip(
                self.poolings, self.blocks
            )):
                g = pool(g)
                if self.use_norm:
                    g = self.pool_norms[i](g)
                g = blk(g)
                if is_last(i) and self.output_trivial:
                    g = self.regular2trivial(g)
                    g = g.tensor  # Extract the raw tensor from GeometricTensor
                if return_features:
                    features.append(g)

        outputs["output"] = g
        if return_features:
            outputs["features"] = features
        return outputs

    def forward(self, x, return_features: bool = False) -> Dict:
        # NOTE: ESCNN expands bias via a matmul (`bias_expansion @ bias`).
        # Under AMP (bfloat16), this can produce a bf16 bias while the input stays fp32,
        # which then crashes in conv2d with "Input type (float) and bias type (BFloat16)".
        # Run the equivariant encoder in full precision when autocast is enabled.
        # if x.is_cuda and torch.is_autocast_enabled():
        #     with torch.autocast(device_type="cuda", enabled=False):
        #         return self._forward_impl(x.float(), return_features=return_features)
        return self._forward_impl(x, return_features=return_features)
    
