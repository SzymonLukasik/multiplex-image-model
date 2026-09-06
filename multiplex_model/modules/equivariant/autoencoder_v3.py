"""Fully-equivariant V3 autoencoder + FRESH-interface adapter.

``FullyEquivariantMultiplexAutoencoderV3`` is ported verbatim from the OLD
``train_masked_model_ddp_ed.py``. ``EquivariantMultiplexAutoencoderV3`` wraps it
in the interface the refactored framework expects (see
``multiplex_model/modules/immuvis.py::MultiplexAutoencoder``): ``forward`` /
``encode`` / ``decode`` speaking plain tensors at the latent boundary,
``get_architecture_config`` / ``load_from_checkpoint`` for checkpointing, and an
optional reduced-resolution mode (``internal_image_size``).
"""

import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import escnn.nn as e2nn

from .equivariant_modules_v2 import (
    EquivariantMultiplexImageEncoder as EquivariantMultiplexImageEncoderV2,
)
from .decoder_v3 import EquivariantMultiplexImageDecoderV3


class FullyEquivariantMultiplexAutoencoderV3(nn.Module):
    """Equivariant encoder (v2) + single-block frequency-schedule decoder (V3).

    Same encoder pipeline as V2 (so existing encoder configs and checkpoints
    are compatible); the decoder is `EquivariantMultiplexImageDecoderV3` —
    one BLConvNeXt block at the bottleneck followed by bilinear-upsample +
    1×1 R2Conv refinement with a max_freq schedule. Designed against the
    OOM cliff and the freq-0 starvation that broke V2's recon (see
    MEMORY_EFFICIENT_EQUIVARIANT_DECODER.md and
    BLANK_RECON_DIAGNOSIS_AND_NEXT_RUNS.md).
    """

    def __init__(
        self,
        num_channels: int,
        encoder_config: Dict,
        decoder_config: Dict,
    ):
        super().__init__()
        self.num_channels = num_channels

        # Encoder: identical to V2 (keep BL-regular all the way through).
        enc_cfg = dict(encoder_config)
        enc_cfg['output_trivial'] = False
        enc_cfg.pop('output_scalars', None)
        self.encoder = EquivariantMultiplexImageEncoderV2(
            num_channels=num_channels, **enc_cfg,
        )

        gspace = self.encoder._gspace
        bl_repr = self.encoder.marker_agnostic_encoder.bl_repr

        hk_scale = (
            encoder_config['hyperkernel_config'].get('stride', 1)
            * decoder_config['hyperkernel_config'].get('stride', 1)
        )
        scaling_factor = hk_scale * 2 ** len(
            encoder_config['ma_layers_blocks']
            + encoder_config['pm_layers_blocks'][:-1]
        )
        input_field_dim = encoder_config['pm_embedding_dims'][-1]

        # Block-flavor flags shared with the encoder (consistent with V2's
        # convention of reading these from encoder_config).
        block_kwargs = dict(
            use_gating=encoder_config.get('use_gating', True),
            use_layerscale=encoder_config.get('use_layerscale', True),
            use_norm=encoder_config.get('use_norm', True),
            layerscale_init=encoder_config.get('layerscale_init', 1e-6),
            gate_bias_init=encoder_config.get('gate_bias_init', 1.0),
        )

        # Decoder-only flags: pop before splatting so they don't collide.
        dec_cfg = dict(decoder_config)
        decoder_only_keys = (
            'in_block_norm', 'max_freq_schedule', 'field_dim_schedule',
            'mean_head_bias_init', 'post_refine_norm',
            'upsample_type', 'upsample_kernel_size', 'upsample_padding',
            'upsample_output_padding', 'norm_act_function',
        )
        decoder_only_kwargs = {
            k: dec_cfg.pop(k) for k in decoder_only_keys if k in dec_cfg
        }

        # V2's `num_blocks` knob doesn't apply to V3 (there is exactly one
        # block at the bottleneck by design). Quietly drop it so older
        # encoder/decoder yamls remain forward-compatible.
        dec_cfg.pop('num_blocks', None)

        self.decoder = EquivariantMultiplexImageDecoderV3(
            input_field_dim=input_field_dim,
            scaling_factor=scaling_factor,
            num_channels=num_channels,
            gspace=gspace,
            bl_repr=bl_repr,
            **block_kwargs,
            **decoder_only_kwargs,
            **dec_cfg,
        )

    def encode_images(
        self, x: torch.Tensor, encoded_indices: torch.Tensor,
        return_features: bool = False,
    ) -> Dict:
        enc_out = self.encoder(
            x, encoded_indices=encoded_indices,
            return_features=return_features,
        )
        outputs = {'output': enc_out['output']}
        if return_features:
            outputs['features'] = enc_out['features']
        return outputs

    def decode_images(
        self, z, decoded_indices: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        return self.decoder(z, decoded_indices, output_size=output_size)

    def forward(
        self,
        x: torch.Tensor,
        encoded_indices: torch.Tensor,
        decoded_indices: torch.Tensor,
        return_features: bool = False,
    ) -> Dict:
        input_size = x.shape[-2:]
        enc_out = self.encode_images(
            x, encoded_indices, return_features=return_features,
        )
        z = enc_out['output']
        recon = self.decode_images(z, decoded_indices, output_size=input_size)
        outputs = {'output': recon}
        if return_features:
            outputs['features'] = enc_out['features']
        return outputs



class EquivariantMultiplexAutoencoderV3(FullyEquivariantMultiplexAutoencoderV3):
    """FRESH-interface adapter around the fully-equivariant V3 autoencoder.

    Adds three things on top of the ported model:

    1. **Plain-tensor latent boundary.** The refactored validation loop pools and
       normalizes ``encode()["output"]`` and feeds it to RankMe, so it must be a
       plain ``torch.Tensor``; the equivariant encoder emits an escnn
       ``GeometricTensor``. ``encode`` returns ``z.tensor`` and stashes the
       ``FieldType`` so ``decode`` can re-wrap it.
    2. **Optional reduced-resolution operation.** When ``internal_image_size`` is
       set, the input is bilinearly downscaled to that size before the escnn core
       and the reconstruction is bilinearly upscaled back to the original size, so
       the loss is computed at the ORIGINAL resolution by the unchanged training
       loop (drop-in comparable to the full-res baselines).
    3. **Checkpointing** via ``get_architecture_config`` / ``load_from_checkpoint``
       matching ``MultiplexAutoencoder``.

    Coupling assumption (satisfied by the train/val loops in this repo):
    ``encode()`` is always called before ``decode()`` for the same batch. ``encode``
    stashes the latent ``FieldType`` and the pre-downscale spatial size that
    ``decode`` needs.
    """

    MODEL_TYPE = "fully_equivariant_v3"

    def __init__(self, num_channels: int, encoder_config: dict, decoder_config: dict):
        encoder_config = dict(encoder_config)
        # migration-only knob; the escnn encoder must not receive it
        self._internal_size = encoder_config.pop("internal_image_size", None)

        # architecture config for checkpointing (keep the knob + model_type)
        self._arch = {
            "num_channels": num_channels,
            "encoder_config": copy.deepcopy(
                {**encoder_config, "internal_image_size": self._internal_size}
            ),
            "decoder_config": copy.deepcopy(dict(decoder_config)),
            "model_type": self.MODEL_TYPE,
        }

        super().__init__(
            num_channels=num_channels,
            encoder_config=encoder_config,
            decoder_config=decoder_config,
        )

        self._latent_type = None   # escnn FieldType, captured on the first encode()
        self._external_hw = None   # (H, W) of the input before downscaling
        self._internal_hw = None   # (H, W) fed to the escnn core (== external if no downscale)

    # ── resolution helpers ────────────────────────────────────────────────
    def _downscale(self, x: torch.Tensor) -> torch.Tensor:
        self._external_hw = tuple(x.shape[-2:])
        if self._internal_size is not None and x.shape[-1] != self._internal_size:
            x = F.interpolate(
                x,
                size=(self._internal_size, self._internal_size),
                mode="bilinear",
                align_corners=False,
            )
        self._internal_hw = tuple(x.shape[-2:])
        return x

    def _upscale(self, recon: torch.Tensor) -> torch.Tensor:
        # recon: [B, C, h, w, O]  ->  [B, C, H_ext, W_ext, O]
        if self._internal_size is None or tuple(recon.shape[2:4]) == self._external_hw:
            return recon
        B, C, h, w, O = recon.shape
        H, W = self._external_hw
        r = recon.permute(0, 1, 4, 2, 3).reshape(B, C * O, h, w)
        r = F.interpolate(r, size=(H, W), mode="bilinear", align_corners=False)
        r = r.reshape(B, C, O, H, W).permute(0, 1, 3, 4, 2).contiguous()
        return r

    # ── FRESH interface ───────────────────────────────────────────────────
    def encode(self, x, encoded_indices, spatial_mask=None, return_features=False):
        # spatial_mask is accepted for interface parity; the equivariant encoder
        # has no learnable mask token, so masked pixels are already zeroed upstream.
        x = self._downscale(x)
        enc = self.encoder(
            x, encoded_indices=encoded_indices, return_features=return_features
        )
        z = enc["output"]                 # escnn GeometricTensor
        self._latent_type = z.type
        outputs = {"output": z.tensor}    # plain tensor for the val latent metrics
        if return_features:
            outputs["features"] = enc["features"]
        return outputs

    def decode(self, latent, decoded_indices):
        if self._latent_type is None:
            raise RuntimeError(
                "decode() was called before encode(); the equivariant adapter needs "
                "encode() to capture the latent FieldType and the input resolution."
            )
        z = e2nn.GeometricTensor(latent, self._latent_type)
        recon = self.decoder(z, decoded_indices, output_size=self._internal_hw)
        return self._upscale(recon)

    def forward(
        self,
        x,
        encoded_indices,
        decoded_indices,
        spatial_mask=None,
        return_features=False,
    ):
        enc = self.encode(
            x, encoded_indices, spatial_mask=spatial_mask, return_features=return_features
        )
        outputs = {"output": self.decode(enc["output"], decoded_indices)}
        if return_features:
            outputs["features"] = enc["features"]
        return outputs

    # ── checkpointing (mirrors MultiplexAutoencoder) ──────────────────────
    def get_architecture_config(self, by_alias: bool = False) -> dict:
        config = copy.deepcopy(self._arch)
        if by_alias:
            config["encoder"] = config.pop("encoder_config")
            config["decoder"] = config.pop("decoder_config")
        return config

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint, map_location=None, model_config=None, strict=True
    ):
        if isinstance(checkpoint, dict):
            data = checkpoint
        else:
            data = torch.load(checkpoint, map_location=map_location)
        cfg = data.get("model_config", model_config)
        if cfg is None:
            raise ValueError(
                "Checkpoint is missing 'model_config'; provide model_config to load "
                "the model (old equivariant checkpoints do not embed it)."
            )
        cfg = dict(cfg)
        cfg.pop("model_type", None)  # not a constructor argument
        model = cls(**cfg)
        model.load_state_dict(data["model_state_dict"], strict=strict)
        return model
