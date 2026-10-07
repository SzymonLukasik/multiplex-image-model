"""
Legacy ImmuVis architecture for loading checkpoints trained with the original
dav3794/multiplex-image-model master branch (multiplex_model/modules/immuvis.py).

Three structural differences vs. multiplex_model/modules.py's MultiplexAutoencoder:

  1. ConvNeXtEncoder(stem=False) uses nn.Identity() as the first norm_pooling
     (no LayerNorm, no stride-2 conv). The pan-marker encoder therefore has no
     param entries under norm_poolings.0 and applies no downsample for stage 0.

  2. MultiplexImageEncoder has `self.norm` (channels-first LayerNorm at
     pm_embedding_dims[0]) instead of GELU between the hyperkernel and the
     pan-marker encoder, plus a final `self.latent_norm` at pm_embedding_dims[-1].

  3. The hyperkernel embedding_dim is fixed to pm_embedding_dims[0] (the
     legacy config never specified `embedding_dim` under `hyperkernel`).

  4. Decoder scaling_factor = 2 ** len(ma_layers_blocks + pm_layers_blocks[:-1])
     (the last pm stage does not contribute a downsample because of (1)).

Hyperkernel and the per-block ConvNext modules are reused from modules.py to
avoid duplication; only the encoder/autoencoder wiring differs.
"""
from typing import Dict, List

import torch
import torch.nn as nn

from .modules import (
    ConvNextBlock,
    Hyperkernel,
    Identity,
    LayerNorm,
    MultiplexImageDecoder,
)


class ConvNeXtEncoderLegacy(nn.Module):
    """ConvNeXt encoder matching the original ImmuVis behavior for stem=False."""

    def __init__(
        self,
        input_channels: int,
        layers_blocks: List[int],
        embedding_dims: List[int],
        stem: bool = True,
    ):
        super().__init__()

        self.norm_poolings = nn.ModuleList()
        if stem:
            stem_layer = nn.Sequential(
                nn.Conv2d(input_channels, embedding_dims[0], kernel_size=2, stride=2, padding=0),
                # nn.Conv2d(input_channels, embedding_dims[0], kernel_size=4, stride=2, padding=1),
                LayerNorm(embedding_dims[0], data_format="channels_first"),
            )
        else:
            stem_layer = nn.Identity()
        self.norm_poolings.append(stem_layer)

        for i, out_dim in enumerate(embedding_dims[1:]):
            input_dim = embedding_dims[i]
            self.norm_poolings.append(
                nn.Sequential(
                    LayerNorm(input_dim, data_format="channels_first"),
                    nn.Conv2d(input_dim, out_dim, kernel_size=2, padding=0, stride=2),
                )
            )

        self.blocks = nn.ModuleList()
        for blocks, dim in zip(layers_blocks, embedding_dims):
            self.blocks.append(nn.Sequential(*[ConvNextBlock(dim) for _ in range(blocks)]))

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict:
        outputs = {}
        features = []
        for norm_pooling, blocks in zip(self.norm_poolings, self.blocks):
            x = norm_pooling(x)
            x = blocks(x)
            if return_features:
                features.append(x)

        outputs["output"] = x
        if return_features:
            outputs["features"] = features
        return outputs


class MultiplexImageEncoderLegacy(nn.Module):
    """Legacy ImmuVis encoder with norm + latent_norm and Identity pm-stem."""

    def __init__(
        self,
        num_channels: int,
        ma_layers_blocks: List[int],
        ma_embedding_dims: List[int],
        hyperkernel_config: Dict,
        pm_layers_blocks: List[int],
        pm_embedding_dims: List[int],
    ):
        super().__init__()

        if len(ma_layers_blocks) == 0:
            self.marker_agnostic_encoder = Identity()
            hyperkernel_input_dim = 1
        else:
            self.marker_agnostic_encoder = ConvNeXtEncoderLegacy(
                input_channels=1,
                layers_blocks=ma_layers_blocks,
                embedding_dims=ma_embedding_dims,
                stem=True,
            )
            hyperkernel_input_dim = ma_embedding_dims[-1]

        hk_cfg = dict(hyperkernel_config)
        # The legacy code fixes embedding_dim to pm_embedding_dims[0]; ignore any
        # value in the YAML so configs from either era still parse.
        hk_cfg.pop("embedding_dim", None)
        hyperkernel_embedding_dim = pm_embedding_dims[0]

        self.hyperkernel = Hyperkernel(
            num_channels=num_channels,
            input_dim=hyperkernel_input_dim,
            embedding_dim=hyperkernel_embedding_dim,
            module_type="encoder",
            **hk_cfg,
        )
        self.norm = LayerNorm(hyperkernel_embedding_dim, data_format="channels_first")

        self.pan_marker_encoder = ConvNeXtEncoderLegacy(
            input_channels=hyperkernel_embedding_dim,
            layers_blocks=pm_layers_blocks,
            embedding_dims=pm_embedding_dims,
            stem=False,
        )
        self.latent_norm = LayerNorm(pm_embedding_dims[-1], data_format="channels_first")

    def forward(
        self,
        x: torch.Tensor,
        encoded_indices: torch.Tensor,
        return_features: bool = False,
    ) -> Dict:
        outputs = {}
        features = []

        B, C, H, W = x.shape
        x = x.reshape(B * C, 1, H, W)
        x = self.marker_agnostic_encoder(x, return_features=return_features)
        if return_features:
            features += x["features"]
        x = x["output"]
        _, E_ma, H_ma, W_ma = x.shape
        x = x.reshape(B, C, E_ma, H_ma, W_ma).reshape(B, C * E_ma, H_ma, W_ma)

        x = self.hyperkernel(x, encoded_indices)
        x = self.norm(x)
        x = self.pan_marker_encoder(x, return_features=return_features)
        if return_features:
            features += x["features"]
        x = x["output"]
        x = self.latent_norm(x)

        outputs["output"] = x
        if return_features:
            outputs["features"] = features
        return outputs


class MultiplexAutoencoderLegacy(nn.Module):
    """Drop-in MultiplexAutoencoder for legacy ImmuVis checkpoints."""

    def __init__(
        self,
        num_channels: int,
        encoder_config: Dict,
        decoder_config: Dict,
    ):
        super().__init__()
        self.latent_dim = encoder_config["pm_embedding_dims"][-1]
        self.num_channels = num_channels

        self.encoder = MultiplexImageEncoderLegacy(
            num_channels=self.num_channels, **encoder_config
        )

        # Legacy formula: the pm encoder's first stage does NOT downsample.
        ma_blocks = encoder_config["ma_layers_blocks"]
        pm_blocks = encoder_config["pm_layers_blocks"]
        scaling_factor = 2 ** len(list(ma_blocks) + list(pm_blocks[:-1]))

        self.decoder = MultiplexImageDecoder(
            input_embedding_dim=self.latent_dim,
            scaling_factor=scaling_factor,
            num_channels=self.num_channels,
            **decoder_config,
        )

    def encode(
        self,
        x: torch.Tensor,
        encoded_indices: torch.Tensor,
        return_features: bool = False,
    ) -> Dict:
        encoding_output = self.encoder(x, encoded_indices, return_features=return_features)
        outputs = {"output": encoding_output["output"]}
        if return_features:
            outputs["features"] = encoding_output["features"]
        return outputs

    def decode(self, x: torch.Tensor, decoded_indices: torch.Tensor) -> torch.Tensor:
        return self.decoder(x, decoded_indices)

    def forward(
        self,
        x: torch.Tensor,
        encoded_indices: torch.Tensor,
        decoded_indices: torch.Tensor,
        return_features: bool = False,
    ) -> Dict:
        encoding_output = self.encode(x, encoded_indices, return_features=return_features)
        x = encoding_output["output"]
        x = self.decode(x, decoded_indices)
        outputs = {"output": x}
        if return_features:
            outputs["features"] = encoding_output["features"]
        return outputs
