"""VirTues-free embedding-inference runner for the equivariant autoencoder.

Drop-in replacement for ``core.embeddings.inference`` that does NOT import
``datasets.multiplex_dataset`` / ``utils.utils`` from a sibling ``../virtues``
checkout. The cell-typing 'patch' runner only needs a dataset exposing
``crop_index``, ``tissue_index`` and ``get_crop(tid, crop_id, preprocess=False)``
-- all of which come straight from the processed dir built by
``python -m gold_standard.data`` (``core/data.py``). No ESM-2 marker
embeddings, no VirTues config.

Usage (run from the celltyping/ project root, with the training venv):

    python -m core.embeddings.inference_equivariant \
        --config ./gold_standard/gs_config.yaml \
        --registry ./core/models/registry.yaml \
        --model equiv_convnext_v2 \
        --scheme patch \
        --batch_size 32
"""

import argparse
import importlib
import inspect
import os
import random

import numpy as np
import pandas as pd
import torch
import yaml

from core.embeddings.patch import generate_embeddings_patch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)


class ProcessedCellDataset:
    """Lightweight stand-in for VirTues' ``MultiplexDataset``.

    Reads the processed dir produced by ``core/data.py``:
      * ``crop_index.csv``   (tissue_id, crop_id, modality, row, col)
      * ``tissue_index.csv`` (tissue_id, split)
      * ``crops/{tissue_id}_{crop_id}.npy``  -> (C, 32, 32) float
      * ``quantiles.csv``    (optional, exposed for hook compatibility)
    """

    def __init__(self, processed_dir: str, split: str | None = "test"):
        self.processed_dir = processed_dir
        self.crops_dir = os.path.join(processed_dir, "crops")

        self.crop_index = pd.read_csv(
            os.path.join(processed_dir, "crop_index.csv")
        )
        self.tissue_index = pd.read_csv(
            os.path.join(processed_dir, "tissue_index.csv")
        )
        if split is not None and "split" in self.tissue_index.columns:
            keep = self.tissue_index[self.tissue_index["split"] == split][
                "tissue_id"
            ].unique()
            self.tissue_index = self.tissue_index[
                self.tissue_index["tissue_id"].isin(keep)
            ].reset_index(drop=True)
            self.crop_index = self.crop_index[
                self.crop_index["tissue_id"].isin(keep)
            ].reset_index(drop=True)

        q_path = os.path.join(processed_dir, "quantiles.csv")
        self.quantiles = (
            pd.read_csv(q_path, index_col=0) if os.path.exists(q_path) else None
        )

    def get_crop(self, tissue_id, crop_id, preprocess=False):
        # preprocess is intentionally ignored: the universal prepare_fn
        # handles preprocessing (matches core/embeddings/patch.py contract).
        path = os.path.join(self.crops_dir, f"{tissue_id}_{crop_id}.npy")
        arr = np.load(path)
        return torch.from_numpy(arr).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Dataset YAML config")
    parser.add_argument("--registry", default="./core/models/registry.yaml",
                        help="Model registry YAML")
    parser.add_argument("--model", required=True,
                        help="Model key in the registry")
    parser.add_argument("--scheme", choices=["patch"], default="patch")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--split", default="test",
                        help="Tissue split to run on (set 'all' for every "
                             "tissue regardless of split column)")
    args = parser.parse_args()

    with open(args.config) as f:
        conf = yaml.safe_load(f)
    with open(args.registry) as f:
        registry = yaml.safe_load(f)

    processed_dir = conf["processed_dir"]
    annotations_path = os.path.join(processed_dir, "sce_annotations.csv")
    out_dir = os.path.join(
        conf["base_path"], "embeddings", f"{args.model}_{args.scheme}"
    )
    os.makedirs(out_dir, exist_ok=True)

    split = None if args.split == "all" else args.split
    dataset = ProcessedCellDataset(processed_dir, split=split)
    print(f"[inference] {len(dataset.tissue_index)} tissues / "
          f"{len(dataset.crop_index)} crops (split={args.split})")

    models_dict = registry.get("models", {})
    if args.model not in models_dict:
        raise ValueError(
            f"Model '{args.model}' not in {args.registry} under 'models:'"
        )
    model_kwargs = models_dict[args.model].copy()
    setup_func_path = model_kwargs.pop("setup_func")
    module_name, func_name = setup_func_path.rsplit(".", 1)
    setup_func = getattr(importlib.import_module(module_name), func_name)

    model_kwargs["scheme"] = args.scheme
    model_kwargs["batch_size"] = args.batch_size
    sig = inspect.signature(setup_func)
    if "dataset_name" in sig.parameters:
        model_kwargs["dataset_name"] = conf["dataset_name"]

    model, prepare_fn, get_channels, compute_fn, device = setup_func(
        **model_kwargs
    )

    generate_embeddings_patch(
        dataset=dataset,
        model=model,
        prepare_fn=prepare_fn,
        get_channels=get_channels,
        compute_fn=compute_fn,
        annotations_path=annotations_path,
        out_dir=out_dir,
        device=device,
        batch_size=args.batch_size,
    )
    print(f"[inference] embeddings written to {out_dir}")


if __name__ == "__main__":
    main()
