"""Equivariant V3 decoder (ported from the OLD monolithic training script
``train_masked_model_ddp_ed.py``).

Contains, unchanged apart from this import header:
  * ``EquivariantPixelShuffleUpsample`` / ``EquivariantTransposedConvUpsample`` —
    the two escnn upsamplers,
  * ``EquivariantMultiplexImageDecoderV3`` — single BLConvNeXt bottleneck block
    followed by frequency-scheduled upsample + 1x1 R2Conv refinement.

The escnn building blocks come from the ported ``equivariant_modules_v2`` module
under the same (V2-aliased) names the original code used.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import escnn
import escnn.nn as e2nn

from .equivariant_modules_v2 import (
    BLConvNeXtBlock as BLConvNeXtBlockV2,
    EquivariantHyperkernel as EquivariantHyperkernelV2,
    EquivariantPixelLN as EquivariantPixelLNV2,
)


class EquivariantPixelShuffleUpsample(nn.Module):
    """Learnable sub-pixel upsampler for BL-regular fields.

    Replaces the "bilinear interpolate + R2Conv refine" pattern with a
    single 1×1 R2Conv that produces A² sub-pixel patterns per low-res
    location, then reshapes channel→spatial. Crucial property: the A²
    sub-pixel values are *learned*, so the output can carry
    high-spatial-frequency content that bilinear interpolation cannot
    produce — addressing the flat-reconstruction failure mode of V3 v3.1.

    Args:
        in_type    : FieldType of input (at low resolution).
        out_type   : FieldType of output (at high resolution). All reps in
                     out_type must have the same repr size (BL-regular
                     usage: [r] * K).
        upscale_factor : integer A; output spatial = A × input spatial.
    """

    def __init__(
        self,
        in_type: e2nn.FieldType,
        out_type: e2nn.FieldType,
        upscale_factor: int = 2,
    ):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.A = int(upscale_factor)
        A2 = self.A * self.A

        out_reps = list(out_type.representations)
        rep_sizes = set(r.size for r in out_reps)
        assert len(rep_sizes) == 1, (
            f"EquivariantPixelShuffleUpsample requires uniform repr size in "
            f"out_type, got sizes={rep_sizes}"
        )
        self.K_out = len(out_reps)
        self.R = out_reps[0].size

        # Expanded FieldType: A² copies of out_type's reps. After the conv,
        # we reshape these A² copies into the new 2×2 sub-pixel grid.
        gspace = out_type.gspace
        expanded_reps = out_reps * A2
        self.expanded_out_type = e2nn.FieldType(gspace, expanded_reps)

        # 1×1 R2Conv at LOW res, generating A²·K_out output fields. ESCNN
        # constructs the equivariant intertwiner basis automatically.
        self.expand_conv = e2nn.R2Conv(
            in_type, self.expanded_out_type,
            kernel_size=1, padding=0, bias=True, initialize=True,
        )

    def forward(self, x: e2nn.GeometricTensor) -> e2nn.GeometricTensor:
        # Apply channel-expansion conv at LOW resolution.
        x = self.expand_conv(x)
        t = x.tensor                          # (N, K_out·A²·R, H, W)
        N, _, H, W = t.shape
        A = self.A
        K_out = self.K_out
        R = self.R

        # Channel layout (from `expanded_reps = out_reps * A²`): the A²
        # copies are stacked as the outer chunk, with K_out reps inside
        # each chunk. So view as (N, A², K_out, R, H, W); split A² → (A, A);
        # permute and reshape to (N, K_out·R, H·A, W·A) with sub-pixels
        # interleaved into the spatial dims.
        t = t.view(N, A, A, K_out, R, H, W)
        # source axes: 0=N, 1=A_h, 2=A_w, 3=K_out, 4=R, 5=H, 6=W
        # target order: (N, K_out, R, H, A_h, W, A_w)
        t = t.permute(0, 3, 4, 5, 1, 6, 2).contiguous()
        t = t.view(N, K_out * R, H * A, W * A)

        return e2nn.GeometricTensor(t, self.out_type)


class EquivariantTransposedConvUpsample(nn.Module):
    """Learnable, *genuinely equivariant* sub-pixel upsampler for steerable fields.

    Tier 3.1 of EQUIVARIANT_DECODER_NEXT_EXPERIMENTS.md. Wraps
    ``escnn.nn.R2ConvTransposed`` (a transposed convolution whose kernel lives in
    the steerable/equivariant basis). Unlike ``EquivariantPixelShuffleUpsample``
    — whose channel→spatial reshape places the A² sub-pixel copies at fixed grid
    positions that do **not** co-rotate with the group action (the dominant
    equivariance break flagged in EQUIVARIANT_DECODER_CONSISTENCY_PLAN.md) — a
    transposed steerable convolution upsamples by a genuine spatial operation, so
    ``T(up(x)) == up(T(x))`` up to the basis band limit, while staying fully
    learnable (recovering the sharpness pixel-shuffle was introduced for).

    For ``upscale_factor = A`` the conv runs with ``stride = A``. The default
    ``kernel_size=4, padding=1, output_padding=0`` gives output spatial exactly
    ``A × input`` for ``A = 2`` — out = (H-1)·2 − 2·1 + (4−1) + 0 + 1 = 2H — with
    symmetric padding and **no** asymmetric ``output_padding`` (a centered
    sampling grid is better for D4 equivariance than configs needing
    output_padding). All conv geometry is overridable; if you change ``A``,
    re-pick kernel/padding so the output stays exactly ``A×``.

    Args:
        in_type        : input FieldType (low resolution).
        out_type       : output FieldType (high resolution). May carry a
                         different field count / max_freq (frequency schedule).
        upscale_factor : integer A; output spatial = A × input spatial.
        kernel_size, padding, output_padding : transposed-conv geometry.
    """

    def __init__(
        self,
        in_type: e2nn.FieldType,
        out_type: e2nn.FieldType,
        upscale_factor: int = 2,
        kernel_size: int = 4,
        padding: int = 1,
        output_padding: int = 0,
    ):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.A = int(upscale_factor)
        self.up_conv = e2nn.R2ConvTransposed(
            in_type, out_type,
            kernel_size=kernel_size,
            stride=self.A,
            padding=padding,
            output_padding=output_padding,
            bias=True,
            initialize=True,
        )

    def forward(self, x: e2nn.GeometricTensor) -> e2nn.GeometricTensor:
        return self.up_conv(x)


# ═══════════════════════════════════════════════════════════════════════════════
# V3 decoder: single-block + frequency schedule (Design B from
# MEMORY_EFFICIENT_EQUIVARIANT_DECODER.md). Heavy lifting at the bottleneck
# (15×15) with ONE BLConvNeXt block; spatial upsample is now done via
# equivariant pixel-shuffle (v3.2; replaces v3.1's bilinear + 3×3 R2Conv).
#
# Concurrently fixes both root causes from
# BLANK_RECON_DIAGNOSIS_AND_NEXT_RUNS.md:
#   - Cause #2 (freq-0 starvation at regular2output): the upstream features
#     have *already* been compressed to mostly-scalar reps by the time we
#     read them, thanks to the freq schedule [2, 1, 0].
#   - Bilinear-only blur (v3.1 diagnosis): pixel-shuffle replaces bilinear
#     with a learned sub-pixel upsample, so the decoder can produce
#     high-spatial-frequency content at full resolution.
# ═══════════════════════════════════════════════════════════════════════════════
class EquivariantMultiplexImageDecoderV3(nn.Module):
    """Single-block fully-equivariant decoder with frequency-schedule upsampling.

    v3.2 (current): pixel-shuffle upsampling. See `EquivariantPixelShuffleUpsample`.

    Architecture (3 upsample stages, 15×15 → 120×120 → cropped to output_size):

        z (BL-regular, K_in fields @ max_freq=2, H_lat × W_lat)
          ─► EquivariantHyperkernel (decoder-mode, intertwiner_basis='full')
          ─► (N=B·C, K_bn · R_bn, H_lat, W_lat) at bottleneck FieldType
          ─► BLConvNeXtBlock (ONE block; the only heavy compute)
          ─► EquivariantPixelLN
          ─► NormNonLinearity
          ─► for i in range(num_stages):
                pixel_shuffle_upsamples[i]: 1×1 R2Conv producing A²·K_out
                  output fields at LOW res; reshape channel→spatial → H·A × W·A
                refine_norms[i]: optional EquivariantPixelLN
                refine_acts[i] : NormNonLinearity
          ─► regular2output: R2Conv 1×1, FieldType (last) → Trivial(num_outputs)
          ─► center-crop to output_size if pixel-shuffle output exceeds it
          ─► (B, C, H_out, W_out, num_outputs)

    The FieldType at each step is built from `bl_regular_representation(f)`
    where `f` runs over `max_freq_schedule`. `field_dim_schedule` controls
    the number of fields at each post-shuffle step. Both default to keeping
    the bottleneck values (i.e. no schedule reduction; equivalent to V2's
    "no schedule, no field-dim reduction" but with only one block total).

    Earlier iterations (kept in the code as commented-out blocks for
    reference) are documented in V3_LOG.md:
      - v3.0: bilinear up → 1×1 R2Conv refine. OOM and flat recon.
      - v3.1: 1×1 R2Conv (pre-bilinear, channel reduction) → bilinear up
              → 3×3 R2Conv (post-bilinear, spatial mixing). Memory fit but
              still flat recon — bilinear's smoothing dominated.
      - v3.2: pixel-shuffle replaces bilinear+refine. Learnable sub-pixel.
    """

    def __init__(
        self,
        input_field_dim: int,        # K_in: number of BL-regular fields from encoder
        decoded_field_dim: int,      # K_bn: number of fields at the bottleneck (after hyperkernel)
        scaling_factor: int,          # total spatial upsample factor (e.g. 8)
        num_channels: int,            # number of marker channels (vocab for hyperkernel embed)
        gspace,                       # SHARED gspace from the encoder
        bl_repr,                      # encoder's BL-regular irrep (max_freq=2)
        hyperkernel_config: Dict,
        max_freq_schedule: Optional[List[int]] = None,
        field_dim_schedule: Optional[List[int]] = None,
        num_outputs: int = 2,
        use_gating: bool = True,
        use_layerscale: bool = True,
        use_norm: bool = True,          # post-block + post-refine PixelLN
        in_block_norm: bool = True,     # PixelLN INSIDE the BLConvNeXt block
        layerscale_init: float = 1e-6,
        gate_bias_init: float = 1.0,
        mean_head_bias_init: float = -2.2,   # logit(0.1) — IMC channel-mean prior
        post_refine_norm: bool = False,      # cheap EquivariantPixelLN between refine_conv and refine_act
        upsample_type: str = 'pixel_shuffle',  # 'pixel_shuffle' (v3.2) | 'transposed_conv' (Tier 3.1, equivariant)
        upsample_kernel_size: int = 4,        # transposed-conv geometry (only used for 'transposed_conv')
        upsample_padding: int = 1,
        upsample_output_padding: int = 0,
        norm_act_function: str = 'n_relu',    # NormNonLinearity gate: 'n_relu' (default, hard deadzone) |
                                              # 'n_softplus' (smooth, no deadzone) | 'n_sigmoid'. The
                                              # ReLU deadzone is the prime suspect for the flat-background
                                              # collapse — small activations get zeroed, reinforcing the
                                              # variance-soaking minimum the vanilla GELU decoder escapes.
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.num_outputs = num_outputs
        self.decoded_field_dim = decoded_field_dim
        self.input_field_dim = input_field_dim
        self.repr_dim_bottleneck = bl_repr.size
        self.num_stages = int(math.log2(scaling_factor))
        assert 2 ** self.num_stages == scaling_factor, \
            f"scaling_factor must be a power of 2, got {scaling_factor}"

        # ── Default schedules ─────────────────────────────────────────────
        # max_freq_schedule[i] is the max_freq AFTER the i-th refine. The
        # bottleneck (pre-upsample) uses bl_repr's max_freq, i.e. the
        # encoder's max_freq=2.
        bottleneck_max_freq = self._infer_max_freq_from_repr(bl_repr)
        if max_freq_schedule is None:
            max_freq_schedule = [bottleneck_max_freq] * self.num_stages
        if field_dim_schedule is None:
            field_dim_schedule = [decoded_field_dim] * self.num_stages
        assert len(max_freq_schedule) == self.num_stages, \
            f"max_freq_schedule must have num_stages={self.num_stages} entries"
        assert len(field_dim_schedule) == self.num_stages, \
            f"field_dim_schedule must have num_stages={self.num_stages} entries"
        for f in max_freq_schedule:
            assert 0 <= f <= bottleneck_max_freq, \
                f"max_freq_schedule entry {f} must be in [0, {bottleneck_max_freq}]; " \
                f"the gspace was built with max_freq={bottleneck_max_freq} so higher " \
                f"frequencies are not representable in its irrep cache."
        # Monotonic-non-increasing schedule is the only sensible one; we don't
        # enforce it (you might want to mix), but warn-by-assert in the common case.

        self.max_freq_schedule = max_freq_schedule
        self.field_dim_schedule = field_dim_schedule

        # ── Build BL-regular reps for each scheduled max_freq ─────────────
        G = gspace.fibergroup
        bl_reprs_per_stage = [
            bl_repr if f == bottleneck_max_freq else G.bl_regular_representation(f)
            for f in max_freq_schedule
        ]

        # ── FieldTypes ────────────────────────────────────────────────────
        self.bottleneck_type = e2nn.FieldType(
            gspace, [bl_repr] * decoded_field_dim,
        )
        stage_types = [
            e2nn.FieldType(gspace, [r] * k)
            for r, k in zip(bl_reprs_per_stage, field_dim_schedule)
        ]
        self.stage_types = stage_types  # for introspection

        # ── Decoder-mode hyperkernel: encoder latent (K_in, max_freq=2) →
        # bottleneck (K_bn, max_freq=2). intertwiner_basis defaults to 'full'
        # so the freq-{1,2} → freq-0 path exists (Cause #2 fix).
        self.channel_embed = EquivariantHyperkernelV2(
            num_channels=num_channels,
            input_fields=input_field_dim,
            output_fields=decoded_field_dim,
            gspace=gspace,
            bl_repr=bl_repr,
            use_bias=hyperkernel_config.get('use_bias', True),
            intertwiner_basis=hyperkernel_config.get('intertwiner_basis', 'full'),
            module_type='decoder',
        )

        # ── ONE BLConvNeXt block at the bottleneck (the only heavy compute) ──
        # Costs ~K·R·expansion=1280·4≈5120 mid-block channels at 15×15.
        # At N=B·C≤200 that's ~370 MB per buffer in bf16 — cheap.
        self.bottleneck_block = BLConvNeXtBlockV2(
            in_type=self.bottleneck_type,
            use_gating=use_gating,
            use_norm=in_block_norm,
            use_layerscale=use_layerscale,
            layerscale_init=layerscale_init,
            gate_bias_init=gate_bias_init,
        )
        self.bottleneck_norm = (
            EquivariantPixelLNV2(self.bottleneck_type, eps=1e-6,
                                 center_scalar=True, affine=True)
            if use_norm else nn.Identity()
        )
        self.bottleneck_act = e2nn.NormNonLinearity(
            self.bottleneck_type, function=norm_act_function)
        self.norm_act_function = norm_act_function

        # ── Per-stage upsample (selectable via `upsample_type`) ───────────
        #   'pixel_shuffle'   (v3.2): a 1×1 R2Conv at low res generates A²·K_out
        #       output fields; a channel→spatial reshape produces A·H × A·W.
        #       Sharp but NOT equivariant — the reshape's sub-pixel placement
        #       does not co-rotate with the group action (the dominant break in
        #       EQUIVARIANT_DECODER_CONSISTENCY_PLAN.md).
        #   'transposed_conv' (Tier 3.1): an escnn R2ConvTransposed with a
        #       steerable basis — learnable *and* equivariant by construction.
        #       See `EquivariantTransposedConvUpsample`.
        if upsample_type not in ('pixel_shuffle', 'transposed_conv'):
            raise ValueError(
                f"upsample_type must be 'pixel_shuffle' or 'transposed_conv', "
                f"got {upsample_type!r}"
            )
        self.upsample_type = upsample_type
        self.upsamples = nn.ModuleList()
        self.refine_norms = nn.ModuleList()
        self.refine_acts = nn.ModuleList()
        prev_type = self.bottleneck_type
        for i in range(self.num_stages):
            target_type = stage_types[i]
            if upsample_type == 'pixel_shuffle':
                self.upsamples.append(
                    EquivariantPixelShuffleUpsample(
                        in_type=prev_type,
                        out_type=target_type,
                        upscale_factor=2,
                    )
                )
            else:  # 'transposed_conv'
                self.upsamples.append(
                    EquivariantTransposedConvUpsample(
                        in_type=prev_type,
                        out_type=target_type,
                        upscale_factor=2,
                        kernel_size=upsample_kernel_size,
                        padding=upsample_padding,
                        output_padding=upsample_output_padding,
                    )
                )
            self.refine_norms.append(
                EquivariantPixelLNV2(target_type, eps=1e-6,
                                     center_scalar=True, affine=True)
                if post_refine_norm else nn.Identity()
            )
            self.refine_acts.append(
                e2nn.NormNonLinearity(target_type, function=norm_act_function))
            prev_type = target_type

        # ── v3.1 refine path (kept commented for reference) ───────────────
        # The bilinear+1×1+3×3 design that we replaced with pixel-shuffle.
        # Diagnosis: bilinear's spatial smoothing dominated the learnable
        # 3×3 R2Conv's contribution at full res, and the decoder produced
        # near-constant per-image outputs after 45k+ steps. See V3_LOG.md.
        #
        # self.refine_convs_pre = nn.ModuleList()
        # self.refine_convs_post = nn.ModuleList()
        # self.refine_norms = nn.ModuleList()
        # self.refine_acts = nn.ModuleList()
        # prev_type = self.bottleneck_type
        # for i in range(self.num_stages):
        #     target_type = stage_types[i]
        #     self.refine_convs_pre.append(
        #         e2nn.R2Conv(prev_type, target_type,
        #                     kernel_size=1, padding=0, bias=True, initialize=True)
        #     )
        #     self.refine_convs_post.append(
        #         e2nn.R2Conv(target_type, target_type,
        #                     kernel_size=3, padding=1, bias=True, initialize=True)
        #     )
        #     self.refine_norms.append(
        #         EquivariantPixelLNV2(target_type, eps=1e-6,
        #                              center_scalar=True, affine=True)
        #         if post_refine_norm else nn.Identity()
        #     )
        #     self.refine_acts.append(e2nn.NormNonLinearity(target_type))
        #     prev_type = target_type

        # ── Final readout to trivial (mean + logvar) ──────────────────────
        trivial_out_type = e2nn.FieldType(
            gspace, [gspace.trivial_repr] * num_outputs,
        )
        self.regular2output = e2nn.R2Conv(
            stage_types[-1], trivial_out_type,
            kernel_size=1, padding=0, bias=True, initialize=True,
        )

        # ── Mean-head bias init (BLANK_RECON_DIAGNOSIS Run 3) ─────────────
        # Init the trivial-output bias so sigmoid(mean) starts near the
        # empirical IMC channel mean (~0.1) rather than 0.5. This is one
        # of the symmetry-breaking interventions that escapes the
        # variance-soaking attractor at initialization.
        self._init_mean_head_bias(mean_head_bias_init)

    @staticmethod
    def _infer_max_freq_from_repr(bl_repr) -> int:
        """Recover the max_freq used to build a bl_regular_representation,
        from its dim. For O(2): R = 2 + 4·max_freq."""
        R = bl_repr.size
        # R = 2 + 4·max_freq  ⇒  max_freq = (R − 2) / 4
        mf = (R - 2) // 4
        assert R == 2 + 4 * mf, (
            f"bl_repr.size={R} is not of the form 2+4·max_freq; "
            f"is this an O(2) BL-regular representation?"
        )
        return mf

    def _init_mean_head_bias(self, mean_bias: float) -> None:
        """Bias the mean output channel so sigmoid(pre-sigmoid output) starts
        near the empirical channel mean. ESCNN's R2Conv stores a bias only on
        trivial-irrep components; for a fully-trivial output FieldType this
        is one parameter per output field (so num_outputs entries: index 0
        is mean, index 1 is logvar)."""
        bias = getattr(self.regular2output, 'bias', None)
        if bias is None:
            # Some ESCNN versions wrap the bias differently; do nothing on
            # those and rely on natural training to break symmetry.
            return
        with torch.no_grad():
            bias.zero_()
            bias[0] = float(mean_bias)
            # logvar bias stays 0 → exp(0) = 1, neither saturating nor too tight

    def forward(
        self,
        z,
        indices: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        z:        BL-regular latent — either a torch.Tensor (B, K_in·R, H, W)
                  or a GeometricTensor wrapping the same.
        indices:  (B, C) marker tokens.
        output_size: target (H_out, W_out) for the final upsample.

        returns:  (B, C, H_out, W_out, num_outputs)
        """
        if isinstance(z, e2nn.GeometricTensor):
            z = z.tensor
        B = z.shape[0]
        C = indices.shape[1]
        K_bn = self.decoded_field_dim
        R_bn = self.repr_dim_bottleneck

        # Equivariant marker embed → (B, C, K_bn·R_bn, H_lat, W_lat)
        x = self.channel_embed(z, indices)
        _, _, _, H_lat, W_lat = x.shape
        N = B * C
        x = x.reshape(N, K_bn * R_bn, H_lat, W_lat)
        x = e2nn.GeometricTensor(x, self.bottleneck_type)

        # ── ONE block at the bottleneck (the only heavy compute) ──────────
        x = self.bottleneck_block(x)
        x = self.bottleneck_norm(x)
        x = self.bottleneck_act(x)

        # ── v3.2 pixel-shuffle upsample cascade ────────────────────────────
        # Per stage:
        #   (1) pixel_shuffle_upsamples[i]: 1×1 R2Conv at LOW res producing
        #       A²·K_out output fields, then reshape channel→spatial. Output
        #       is at A·H_in × A·W_in with FieldType stage_types[i]. Both
        #       the FieldType change (freq schedule) and the spatial
        #       upscaling happen here in one learnable op.
        #   (2) refine_norms[i]: optional EquivariantPixelLN.
        #   (3) refine_acts[i] : NormNonLinearity at stage_types[i].
        # All stages use upscale_factor=2, so total spatial is 2^num_stages
        # × H_lat. For H_lat=15 and num_stages=3 that's 120; we center-crop
        # to output_size below if needed.
        for i in range(self.num_stages):
            x = self.upsamples[i](x)
            x = self.refine_norms[i](x)
            x = self.refine_acts[i](x)

        # ── v3.1 forward (kept commented for reference) ────────────────────
        # for i in range(self.num_stages):
        #     # (1) Channel reduction at low res
        #     x = self.refine_convs_pre[i](x)
        #     # (2) Bilinear upsample (raw tensor, then re-wrap with same type)
        #     t = x.tensor
        #     if i == self.num_stages - 1 and output_size is not None:
        #         t = F.interpolate(t, size=output_size, mode='bilinear',
        #                           align_corners=False)
        #     else:
        #         t = F.interpolate(t, scale_factor=2, mode='bilinear',
        #                           align_corners=False)
        #     x = e2nn.GeometricTensor(t, x.type)
        #     # (3) Learnable spatial mixing at high res, reduced channels
        #     x = self.refine_convs_post[i](x)
        #     x = self.refine_norms[i](x)
        #     x = self.refine_acts[i](x)

        # ── Final per-pixel readout to trivial (mean + logvar) ────────────
        x = self.regular2output(x).tensor    # (N, num_outputs, H_full, W_full)

        # ── Center-crop to output_size if pixel-shuffle overshot ──────────
        # Pixel-shuffle factor 2 per stage produces H_lat · 2^num_stages,
        # which may exceed the original input H (e.g. 15·8 = 120 ≠ 113). We
        # center-crop here so the loss compares against the original image
        # extent. The model learns to put useful content in the central
        # region (margin = (H_full − H_out) / 2 on each side; tiny ratio).
        if output_size is not None:
            _, _, H_full, W_full = x.shape
            H_target, W_target = int(output_size[0]), int(output_size[1])
            if H_full > H_target:
                top = (H_full - H_target) // 2
                x = x[:, :, top:top + H_target, :]
            if W_full > W_target:
                left = (W_full - W_target) // 2
                x = x[:, :, :, left:left + W_target]

        _, _, H_out, W_out = x.shape
        x = x.reshape(B, C, self.num_outputs, H_out, W_out)
        x = x.permute(0, 1, 3, 4, 2)         # (B, C, H_out, W_out, num_outputs)
        return x


