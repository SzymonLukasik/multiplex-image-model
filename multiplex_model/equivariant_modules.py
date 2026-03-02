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
from typing import Dict, Type, Callable, Literal
from typing_extensions import Literal
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
    LN-ish normalization for irrep-split FieldTypes:

    - computes per-field magnitude m_f(b,h,w) = ||x_f(b,h,w)||_2 over irrep components
    - s(b,h,w) = mean_f m_f(b,h,w)
    - rescales each field: x_f <- x_f / (s + eps)

    Optional:
    - center_trivial: subtract per-pixel mean across *trivial scalar fields only*
    - affine: learn one gamma per field (equivariant) and beta only for trivial scalar fields
    """
    def __init__(
        self,
        in_type: e2nn.FieldType,
        eps: float = 1e-6,
        center_trivial: bool = True,
        affine: bool = True,
    ):
        super().__init__()
        self.in_type = in_type
        self.out_type = in_type
        self.eps = eps
        self.center_trivial = center_trivial
        self.affine = affine

        # Precompute slices for each field in the concatenated tensor
        self._fields = []  # list of dicts: {start, end, size, is_trivial, trivial_idx}
        start = 0
        trivial_field_indices = []
        for i, r in enumerate(in_type.representations):
            size = r.size
            end = start + size
            is_trivial = (r == in_type.gspace.trivial_repr)  # true trivial 1D scalar
            if is_trivial:
                trivial_field_indices.append(i)
            self._fields.append(
                dict(start=start, end=end, size=size, is_trivial=is_trivial)
            )
            start = end

        self.n_fields = len(self._fields)
        self.trivial_field_indices = trivial_field_indices
        self.n_trivial = len(trivial_field_indices)

        if affine:
            # One gamma per field copy; applies to all components inside the field (equivariant).
            self.gamma = nn.Parameter(torch.ones(self.n_fields))
            # Beta only for trivial scalar fields (equivariant).
            if self.n_trivial > 0:
                self.beta_trivial = nn.Parameter(torch.zeros(self.n_trivial))
            else:
                self.beta_trivial = None

    def forward(self, x: e2nn.GeometricTensor) -> e2nn.GeometricTensor:
        xt = x.tensor  # [B, C, H, W]
        B, C, H, W = xt.shape

        # Work on a copy because we may center trivial scalars
        y = xt

        # 1) Optional: center ONLY trivial scalar fields per pixel
        if self.center_trivial and self.n_trivial > 0:
            # gather trivial scalars into [B, n_trivial, H, W]
            triv = []
            for fi in self.trivial_field_indices:
                f = self._fields[fi]
                # size is 1 for trivial repr
                triv.append(y[:, f["start"]:f["end"], :, :])
            triv_stack = torch.cat(triv, dim=1)  # [B, n_trivial, H, W]
            triv_mean = triv_stack.mean(dim=1, keepdim=True)  # [B, 1, H, W]

            # subtract mean from each trivial scalar channel
            y = y.clone()
            for fi in self.trivial_field_indices:
                f = self._fields[fi]
                y[:, f["start"]:f["end"], :, :] -= triv_mean

        # 2) Compute per-pixel scale s(b,h,w) = mean_f ||field_f||_2
        # Do it as a running sum to avoid stacking huge tensors.
        sum_mag = None
        for f in self._fields:
            sl = y[:, f["start"]:f["end"], :, :]  # [B, size, H, W]
            if f["size"] == 1:
                mag = sl.abs()  # [B, 1, H, W]
            else:
                # L2 norm across irrep components (size dimension)
                mag = torch.linalg.vector_norm(sl, ord=2, dim=1, keepdim=True)  # [B,1,H,W]

            sum_mag = mag if sum_mag is None else (sum_mag + mag)

        s = sum_mag / float(self.n_fields)  # [B, 1, H, W]
        denom = s + self.eps

        # 3) Rescale each field by the same per-pixel scalar denom
        out = y.clone()
        for i, f in enumerate(self._fields):
            sl = out[:, f["start"]:f["end"], :, :]  # [B, size, H, W]
            sl = sl / denom

            # 4) Optional equivariant affine
            if self.affine:
                sl = sl * self.gamma[i]
                if f["is_trivial"] and self.beta_trivial is not None:
                    # map i -> trivial index
                    t_idx = self.trivial_field_indices.index(i)
                    sl = sl + self.beta_trivial[t_idx]

            out[:, f["start"]:f["end"], :, :] = sl

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


class EquivariantMultiplexImageEncoder(nn.Module):
    """Encoder backbone for encoding multiplex images."""

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
            use_gating: bool = True,
            use_layerscale: bool = True,
            layerscale_init: float = 1e-6,
            gate_bias_init: float = 1.0,
            pool_use_act: bool = True,
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

        # channel-agnostic part
        self.marker_agnostic_encoder = EquivariantConvNeXtEncoder(
            input_channels=1,
            layers_blocks=ma_layers_blocks,
            embedding_dims=ma_embedding_dims,
            maximum_frequency=maximum_frequency,
            latent_nonlinearity=latent_nonlinearity,
            use_gating=use_gating,
            use_layerscale=use_layerscale,
            layerscale_init=layerscale_init,
            gate_bias_init=gate_bias_init,
            pool_use_act=pool_use_act,
        )

        self.hyperkernel = Hyperkernel(
            num_channels=num_channels,
            input_dim=ma_embedding_dims[-1],
            module_type='encoder',
            **hyperkernel_config
        )

        self.act = nn.GELU()

        # pan-marker part
        self.pan_marker_encoder = EquivariantConvNeXtEncoder(
            input_channels=self.hyperkernel.embedding_dim,
            layers_blocks=pm_layers_blocks,
            embedding_dims=pm_embedding_dims,
            include_stem=False,
            maximum_frequency=maximum_frequency,
            latent_nonlinearity=latent_nonlinearity,
            use_gating=use_gating,
            use_layerscale=use_layerscale,
            layerscale_init=layerscale_init,
            gate_bias_init=gate_bias_init,
            pool_use_act=pool_use_act,
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
        x = x['output']
        _, E_ma, H_ma, W_ma = x.shape
        x = x.reshape(B, C, E_ma, H_ma, W_ma).reshape(B, C * E_ma, H_ma, W_ma)

        x = self.hyperkernel(x, encoded_indices)
        
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
        self.latent_dim = encoder_config['pm_embedding_dims'][-1]
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

        scaling_factor = 2 ** len(encoder_config['ma_layers_blocks'] + encoder_config['pm_layers_blocks'])
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
        use_layerscale: bool = True,
        layerscale_init: float = 1e-6,
        gate_bias_init: float = 1.0,   # >0 starts gates ~open
    ):
        super().__init__()

        self.use_grn = use_grn
        self.use_gating = use_gating
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


        # self.norm = EquivariantPixelLN(
        #     in_type,
        #     eps=1e-6,
        #     center_trivial=True,   # matches “center only trivial scalars”
        #     affine=True,
        # )
        # self.norm = e2nn.IIDBatchNorm2d(in_type, affine=True, track_running_stats=False)
        # self.norm = e2nn.FieldNorm(in_type, eps=1e-6, affine=True)
        # self.norm = e2nn.GNormBatchNorm(in_type, affine=True)
        
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
        # y = self.norm(y)

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
        use_layerscale: bool = True,
        layerscale_init: float = 1e-6,
        gate_bias_init: float = 1.0,
        latent_nonlinearity="none",
        pool_use_act: bool = True,
    ):
        super().__init__()
        self.latent_nonlinearity = latent_nonlinearity
        self.pool_use_act = pool_use_act

        # ───────────────────────────────────────────────────────────────
        # 0.  Group bookkeeping
        # ───────────────────────────────────────────────────────────────
        self.max_freq  = maximum_frequency
        # self.r2_act    = escnn.gspaces.rot2dOnR2(N=-1, maximum_frequency=maximum_frequency)        # SO(2)
        self.r2_act    = escnn.gspaces.flipRot2dOnR2(N=-1, maximum_frequency=maximum_frequency)        # SO(2) - axis zero was throwing
        # self.r2_act    = e2cnn.gspaces.Rot2dOnR2(N=-1, maximum_frequency=maximum_frequency)        # SO(2)
        self.G        = self.r2_act.fibergroup
        bl_repr        = self.G.bl_regular_representation(self.max_freq)  # dim = 1+2*max_freq
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
                # identity shortcut if the user opts out of a stem
                self.poolings.append(
                    nn.Sequential(
                        e2nn.R2Conv(
                            prev_type, out_type,
                            kernel_size=3, # was 3x3
                            stride=2,
                            padding=1, # change wrt 74, 75
                            bias=True,
                            initialize=True, # TODO: set to True if you want to initialize
                        ),
                        # e2nn.IIDBatchNorm2d(out_type, affine=True, track_running_stats=False),
                        # e2nn.FieldNorm(out_type, eps=1e-5, affine=True),
                    )
                )
            else:
                # print(f"Pooling from {prev_type} to {out_type}")
                self.poolings.append(
                    # nn.Sequential(
                    #     e2nn.PointwiseAvgPoolAntialiased(
                    #         prev_type,
                    #         sigma=0.5, # 0.66,
                    #         stride=2,
                    #         padding = 0,
                    #         # padding=0,  # no padding, like ConvNeXt
                    #         # initialize=True,  # TODO: set to True if you want to initialize
                    #     ),
                    #     # e2nn.PointwiseAvgPool2D(
                    #     #     prev_type,
                    #     #     kernel_size=2,
                    #     # ),
                    #     e2nn.R2Conv(                   # channel *expander*
                    #         prev_type, out_type,
                    #         kernel_size=1, stride=1, padding=0,
                    #         bias=True, initialize=True
                    #     )
                    # )
                    # nn.Sequential(
                    #     e2nn.PointwiseAvgPoolAntialiased(prev_type, sigma=0.66, stride=2),
                    #     e2nn.R2Conv(
                    #         prev_type,
                    #         out_type,
                    #         kernel_size=3,
                    #         stride=1,
                    #         padding=0,
                    #         bias=True,
                    #         initialize=True,  # TODO: set to True if you want to initialize
                    #     )
                    # )

                    
                    # nn.Sequential(
                    #     # AsymmetricPad2d(prev_type, (1, 0, 1, 0)),  # pad right and bottom by 1
                    #     AsymmetricPad2d(prev_type, (0, 1, 0, 1)),  # pad right and bottom by 1
                    #     e2nn.R2Conv(
                    #         prev_type,
                    #         out_type,
                    #         kernel_size=3,
                    #         stride=2,
                    #         padding=1,  # no padding, like ConvNeXt
                    #         bias=True,
                    #         initialize=True,  # TODO: set to True if you want to initialize
                    #     ),
                    #     # CropRightBottom(out_type, cropright=1, cropbottom=1)  if idx > 0 else nn.Identity()
                    # )
                    nn.Sequential(
                        e2nn.R2Conv(
                            prev_type, out_type,
                            kernel_size=3, # was 3x3
                            stride=2,
                            padding=1, # change wrt 74, 75
                            bias=True,
                            initialize=True, # TODO: set to True if you want to initialize
                        ),
                        # e2nn.FieldNorm(out_type, eps=1e-5, affine=True),
                    )
                    # this is what the 74, and 75 versions had
                    # e2nn.R2Conv(
                    #     prev_type, out_type,
                    #     kernel_size=3,
                    #     stride=2,
                    #     padding=1,
                    #     bias=True,
                    #     initialize=True, # TODO: set to True if you want to initialize
                    # )
                )
            prev_type = out_type

        # ------------------------------------------------------------------
        # 3.  Stage-wise NormNonLinearities (GELU analogue) and blocks
        # ------------------------------------------------------------------
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
                    use_layerscale=use_layerscale,
                    layerscale_init=layerscale_init,
                    gate_bias_init=gate_bias_init,
                )
                for t, n in zip(stage_types, layers_blocks)
            ]
        )

        self.out_type = self.blocks[-1].out_type  # last stage’s FieldType

        self.regular2trivial = Regular2Trivial(
            in_type=self.out_type,
            n_scalars=embedding_dims[-1]
        )

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

    def _forward_impl(self, x: torch.Tensor, return_features: bool = False) -> Dict:
        outputs = {}
        features = []

        g = e2nn.GeometricTensor(x, self.input_type)
        if self.pool_use_act:
            for i, (pool, act, blk) in enumerate(zip(self.poolings, self.acts, self.blocks)):
                g = act(pool(g))
                g = blk(g)
                if i == len(self.blocks) - 1:
                    g = self.regular2trivial(g)
                    g = g.tensor  # Extract the raw tensor from GeometricTensor
                    # g = self._postprocess_latent(g)
                if return_features:
                    features.append(g)
        else:
            for i, (pool, blk) in enumerate(zip(self.poolings, self.blocks)):
                g = pool(g)
                g = blk(g)
                if i == len(self.blocks) - 1:
                    g = self.regular2trivial(g)
                    g = g.tensor  # Extract the raw tensor from GeometricTensor
                    # g = self._postprocess_latent(g)
                if return_features:
                    features.append(g)

        outputs["output"] = g
        if return_features:
            outputs["features"] = features
        return outputs

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict:
        # NOTE: ESCNN expands bias via a matmul (`bias_expansion @ bias`).
        # Under AMP (bfloat16), this can produce a bf16 bias while the input stays fp32,
        # which then crashes in conv2d with "Input type (float) and bias type (BFloat16)".
        # Run the equivariant encoder in full precision when autocast is enabled.
        # if x.is_cuda and torch.is_autocast_enabled():
        #     with torch.autocast(device_type="cuda", enabled=False):
        #         return self._forward_impl(x.float(), return_features=return_features)
        return self._forward_impl(x, return_features=return_features)
    
