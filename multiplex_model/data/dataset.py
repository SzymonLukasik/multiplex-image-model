import os
import random
import hashlib
from glob import glob
from typing import Literal

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset, Sampler

from .transforms import (
    OutlierPruning,
    Preprocessing,
    Denoising,
    Scaling,
    Normalization,
    TorchvisionTransform,
    Pipeline,
    Identity,
)

def _subsample_paths(
    paths: list[str], fraction: float, seed: int, dataset: str
) -> list[str]:
    """Deterministically keep `fraction` of one dataset's images.

    Subsetting happens PER DATASET, not across the pooled list, so shrinking the
    training set never removes a panel outright. That matters here: the model is
    panel-conditioned through the marker tokenizer and hyperkernel, so dropping a
    dataset would remove marker combinations and confound "less data" with "fewer
    markers seen".

    The RNG is seeded from (seed, sha256(dataset)) rather than from position in
    the dataset list, so a given dataset's subset is identical no matter which
    other datasets are configured, and `fraction` is nested: a larger fraction is
    not guaranteed to be a superset, but a fixed fraction is reproducible across
    runs and machines. `paths` must already be sorted - glob order is filesystem
    dependent and would otherwise silently vary between runs.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"subset_fraction must be in (0, 1], got {fraction}")
    if fraction == 1.0 or not paths:
        return paths
    ds_hash = int.from_bytes(hashlib.sha256(dataset.encode()).digest()[:8], "big")
    rng = np.random.default_rng([seed, ds_hash])
    keep = max(1, int(round(fraction * len(paths))))
    idx = np.sort(rng.choice(len(paths), size=keep, replace=False))
    return [paths[i] for i in idx]


class MultiplexDataset(Dataset):
    def __init__(
        self,
        panels_config: dict,
        split: str,
        marker_tokenizer: dict[str, int],
        unsupported_marker_behavior: Literal["error", "drop"] = "error",
        transform=None,
        preprocessing_func: Literal["arcsinh", "log1p"] | None = "arcsinh",
        scaling_func: Literal["minmax", "percentile", "global_clip"] | None = "percentile",
        global_scaling_bound: float = 5.0,
        denoising_func: Literal["median", "gaussian", "butterworth"] | None = "butterworth",
        normalization_func: Literal["zscore_ds"] | None = None,
        outlier_pruning_func: Literal["percentile_clip"] | None = None,
        operation_order: list[str] = [
            "transform",
            "preprocessing",
            "denoising",
            "scaling",
        ],
        file_extension: Literal["tiff", "npy"] = "tiff",
        preprocessing_kwargs: dict = {},
        denoising_kwargs: dict = {},
        scaling_kwargs: dict = {},
        normalization_kwargs: dict = {},
        outlier_pruning_kwargs: dict = {},
        subset_fraction: float | None = None,
        subset_seed: int = 0,
    ):
        """Dataset for loading multiplex images from multiple panels.
        The order of operations is given by `operation_order`.

        Args:
            panels_config (dict): Configuration dictionary for panels.
            split (str): Name of the data split (e.g., 'train', 'val', 'test').
            marker_tokenizer (dict[str, int]): Tokenizer for marker names.
            unsupported_marker_behavior (Literal['error', 'drop'], optional): Behavior when encountering unsupported markers. Defaults to 'error'.
                `error`: Raises an error if unsupported markers are found.
                `drop`: Drops unsupported markers and the corresponding channels and continues processing.
            transform (_type_, optional): Transform to be applied to the images. Defaults to None.
            preprocessing_func (Literal['arcsinh', 'log1p'], optional): Function to use for preprocessing. 
                Skips preprocessing if None. Defaults to 'arcsinh'.
            denoising_func (Literal['median', 'gaussian', 'butterworth'], optional): Function to use for denoising. 
            Skips denoising if None. Defaults to 'butterworth'.
            scaling_func (Literal['minmax', 'percentile', 'global_clip'], optional): Function to use for scaling. Skips scaling if None. Defaults to 'percentile'.
            global_scaling_bound (float, optional): Global upper bound for scaling if `global_clip` is chosen as `scaling_func`. Defaults to 5.0.
            normalization_func (Literal['zscore_ds'], optional): Function to use for normalization. Skips normalization if None. Defaults to 'zscore_ds'.
            outlier_pruning_func (Literal['percentile_clip'], optional): Function to use for outlier pruning. Skips outlier pruning if None. Defaults to 'percentile_clip'.
            operation_order (list[str], optional): Order of operations to be applied.
                Defaults to ['transform', 'preprocessing', 'denoising', 'scaling', 'normalization'].
            file_extension (Literal['tiff', 'npy'], optional): File extension of the images. Defaults to 'tiff'.
            preprocessing_kwargs (dict, optional): Additional keyword arguments for preprocessing function. Defaults to {}.
            denoising_kwargs (dict, optional): Additional keyword arguments for denoising function. Defaults to {}.
            scaling_kwargs (dict, optional): Additional keyword arguments for scaling function. Defaults to {}.
            normalization_kwargs (dict, optional): Additional keyword arguments for normalization function. Defaults to {}.
            outlier_pruning_kwargs (dict, optional): Additional keyword arguments for outlier pruning function. Defaults to {}.
        """
        assert "paths" in panels_config, (
            "Panels config must have 'paths' attribute with paths of splits of the data."
        )
        assert split in panels_config["paths"], (
            f"Panels config must have '{split}' attribute with data path."
        )
        assert "datasets" in panels_config, (
            "Panels config must have 'datasets' attribute with subdirectories."
        )
        assert "markers" in panels_config, (
            "Panels config must have 'markers' attribute with channel IDs."
        )

        self.ds_markers = panels_config["markers"]

        # Load tokenized channel IDs and scan for markers not present in the tokenizer
        self.channel_ids: dict[str, torch.Tensor] = {}
        self.unsupported_markers_per_ds: dict[str, list[str]] = {}

        for dataset in panels_config["datasets"]:
            unsupported_markers = []
            tokenized_markers = []
            for marker in panels_config["markers"][dataset]:
                marker_token = marker_tokenizer.get(marker, -1)  # Use -1 for unsupported markers
                tokenized_markers.append(marker_token)
                if marker_token == -1:
                    unsupported_markers.append(marker)

            self.channel_ids[dataset] = torch.tensor(tokenized_markers, dtype=torch.long)
    
            if len(unsupported_markers) > 0:
                self.unsupported_markers_per_ds[dataset] = unsupported_markers


        if self.unsupported_markers_per_ds:
            msg = f"Unsupported markers found in the panels config (dataset: {{markers}}): {self.unsupported_markers_per_ds}."
            if unsupported_marker_behavior == "error":
                raise ValueError(
                    f"{msg} Provided tokenizer does not recognize these markers. Set unsupported_marker_behavior='drop' to ignore them."
                )

            print(
                f"{msg} These markers will be dropped from the dataset."
            )

        # Load image paths for each dataset
        img_path = panels_config["paths"][split]
        self.imgs = []  # tuples of (img_path, dataset)
        for dataset in panels_config["datasets"]:
            tiffs = sorted(
                glob(os.path.join(img_path, dataset, "imgs", f"*.{file_extension}"))
            )
            if subset_fraction is not None:
                tiffs = _subsample_paths(tiffs, subset_fraction, subset_seed, dataset)
            self.imgs.extend([(tiff, dataset) for tiff in tiffs])

        if subset_fraction is not None:
            # Fingerprint the exact selection so two runs can be PROVEN to have
            # trained on the same images, rather than assumed to have.
            digest = hashlib.sha256(
                "\n".join(os.path.basename(t) for t, _ in self.imgs).encode()
            ).hexdigest()[:16]
            print(
                f"[MultiplexDataset:{split}] subset_fraction={subset_fraction} "
                f"seed={subset_seed} -> {len(self.imgs)} images, fingerprint={digest}"
            )

        # Transformations declaration
        ds_percentiles = panels_config.get("clip_limits", None)
        ds_marker_stats = panels_config.get("marker_stats", None)

        self.preprocess = (
            Preprocessing(preprocessing_func, **preprocessing_kwargs)
            if preprocessing_func
            else Identity()
        )
        self.denoise = (
            Denoising(denoising_func, **denoising_kwargs)
            if denoising_func
            else Identity()
        )
        self.scale = (
            Scaling(
                scaling_func, ds_percentiles, global_scaling_bound, **scaling_kwargs
            )
            if scaling_func
            else Identity()
        )
        self.norm = (
            Normalization(normalization_func, ds_marker_stats, **normalization_kwargs)
            if normalization_func
            else Identity()
        )
        self.outlier_pruning = (
            OutlierPruning(outlier_pruning_func, **outlier_pruning_kwargs)
            if outlier_pruning_func
            else Identity()
        )
        self.transform = TorchvisionTransform(transform) if transform else Identity()

        self.pipeline = Pipeline(
            transforms={
                "preprocessing": self.preprocess,
                "denoising": self.denoise,
                "scaling": self.scale,
                "normalization": self.norm,
                "transform": self.transform,
                "outlier_pruning": self.outlier_pruning,
            },
            operation_order=operation_order,
        )

        self.file_extension = file_extension
        self.read_file_func = (
            tifffile.imread if self.file_extension == "tiff" else np.load
        )
    
    def _prune_unsupported_markers(
        self,
        img,
        channel_ids,
        marker_names,
        dataset,
    ) -> tuple[np.ndarray, torch.Tensor, list[str]]:
        """Remove channels corresponding to unsupported markers."""
        if dataset in self.unsupported_markers_per_ds:
            supported_channel_mask = channel_ids != -1  # Mask for supported channels
            img = img[supported_channel_mask]
            channel_ids = channel_ids[supported_channel_mask]
            marker_names = [
                marker for marker, is_supported in zip(marker_names, supported_channel_mask) if is_supported
            ]

        return img, channel_ids, marker_names

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, dataset = self.imgs[idx]
        channel_ids = self.channel_ids[dataset]
        marker_names = self.ds_markers[dataset]

        img = self.read_file_func(img_path)
        img, channel_ids, marker_names = self._prune_unsupported_markers(
            img, channel_ids, marker_names, dataset
        )
        img = self.pipeline(img, dataset=dataset, marker_names=marker_names)

        img = torch.tensor(img)

        return img, channel_ids, dataset, img_path


class PanelBatchSampler(Sampler):
    """Sampler that yields batches of indices grouped by panels."""

    def __init__(self, dataset, batch_size, shuffle=True, batches_per_epoch=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        # Fixes the number of batches per epoch independently of dataset size.
        # Used by the data-fraction ablation: with it, a 10% run does the same
        # number of OPTIMISER UPDATES per epoch as the 100% run (it just revisits
        # images more often), so "epoch" means the same thing at every fraction
        # and the epoch budget, validation cadence and checkpoint cadence stay
        # comparable. Without it, matching updates would mean running 10x the
        # epochs, and validation - which always runs on the FULL val set - would
        # come to dominate the cost.
        self.batches_per_epoch = batches_per_epoch

        # Group indices by panel
        self.panel_to_indices = {}
        for idx, (_, panel_idx) in enumerate(dataset.imgs):
            if panel_idx not in self.panel_to_indices:
                self.panel_to_indices[panel_idx] = []
            self.panel_to_indices[panel_idx].append(idx)

        # Convert to list of (panel, indices) pairs for easier random selection
        self.panels = list(self.panel_to_indices.keys())

        self.epoch_batches = []  # Store batches for an epoch
        self._generate_batches()  # Prepare the first epoch

    def _generate_batches(self):
        """Build one epoch of batches.

        Without ``batches_per_epoch`` this is one pass: every sample used exactly
        once. With it, passes are concatenated (each reshuffled independently)
        until the target count is reached, then truncated - so a short dataset is
        revisited rather than producing a short epoch.
        """
        batches = self._one_pass()
        if self.batches_per_epoch is not None:
            while len(batches) < self.batches_per_epoch:
                batches.extend(self._one_pass())
            batches = batches[: self.batches_per_epoch]
        self.epoch_batches = batches

    def _one_pass(self):
        """One pass over the data: each sample used exactly once."""
        epoch_batches = []

        # Shuffle panels if needed
        if self.shuffle:
            random.shuffle(self.panels)

        for panel in self.panels:
            indices = self.panel_to_indices[panel]

            # Shuffle indices within the panel if needed
            if self.shuffle:
                random.shuffle(indices)

            # Split indices into batches of batch_size
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                epoch_batches.append(batch)

        # Shuffle the final batch order for diversity
        if self.shuffle:
            random.shuffle(epoch_batches)
        return epoch_batches

    def __iter__(self):
        """Yield batches, ensuring all images are used exactly once per epoch."""
        for batch in self.epoch_batches:
            yield batch
        self._generate_batches()  # Prepare for next epoch

    def __len__(self):
        """Return number of batches per epoch."""
        return len(self.epoch_batches)
