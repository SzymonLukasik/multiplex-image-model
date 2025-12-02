"""Baseline clustering-based variance analysis for inpainted markers.

This script scans all datasets defined in a panel config (except nsclc2-panel1),
clusters pixels using the intersection of Immucan Panel 1 markers, and reports
within-cluster variances for both the intersecting markers and the remaining
markers. It is meant as a baseline for virtual staining performance.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from ruamel.yaml import YAML
from sklearn.cluster import KMeans
from tqdm import tqdm
import tifffile
import matplotlib.pyplot as plt

from multiplex_model.data import DatasetFromTIFF, TestCrop
from src_from_rudy.constants import PANEL_1_MARKER_NAMES


@dataclass
class Sample:
    image: np.ndarray  # (C, H, W) after preprocessing
    dataset: str
    path: str


def load_yaml(path: str) -> Mapping:
    return YAML(typ="safe").load(open(path, "r"))


def infer_crop_size(config: Mapping, explicit: Optional[int]) -> Optional[int]:
    if explicit is not None:
        return explicit
    default_size = config.get("input_image_size")
    if isinstance(default_size, (list, tuple)) and default_size:
        return int(default_size[0])
    return None


def build_dataset_generator(
    panel_config: Mapping,
    tokenizer: Mapping,
    split: str,
    crop_size: Optional[int],
    dataset_name: str,
    use_butterworth: bool,
    use_median: bool,
    use_minmax: bool,
    allow_missing: bool,
    max_images: Optional[int],
    rng: np.random.Generator,
) -> Tuple[Iterable[Sample], int]:
    transform = TestCrop(crop_size) if crop_size is not None else None
    dataset = DatasetFromTIFF(
        panels_config=panel_config,
        split=split,
        marker_tokenizer=tokenizer,
        transform=transform,
        use_median_denoising=use_median,
        use_butterworth_filter=use_butterworth,
        use_minmax_normalization=use_minmax,
        use_clip_normalization=not use_minmax,
    )
    dataset.imgs = [(path, ds) for path, ds in dataset.imgs if ds == dataset_name]
    if max_images is not None and len(dataset.imgs) > max_images:
        indices = rng.permutation(len(dataset.imgs))[:max_images]
        dataset.imgs = [dataset.imgs[i] for i in indices]
    if not dataset.imgs and not allow_missing:
        raise FileNotFoundError(f"No images found for dataset '{dataset_name}' in split '{split}'")
    total_images = len(dataset.imgs)

    def _gen() -> Iterable[Sample]:
        for idx in range(len(dataset)):
            img, _channel_ids, ds_name, img_path = dataset[idx]
            yield Sample(image=np.asarray(img, dtype=np.float32), dataset=ds_name, path=img_path)

    return _gen(), total_images


def center_crop_mask(mask: np.ndarray, size: int) -> np.ndarray:
    h, w = mask.shape[-2], mask.shape[-1]
    top = (h - size) // 2
    left = (w - size) // 2
    return mask[top : top + size, left : left + size]


def load_mask_for_sample(sample: Sample, mask_folder: str, crop_size: Optional[int]) -> Optional[np.ndarray]:
    if f"{os.sep}imgs{os.sep}" in sample.path:
        mask_path = sample.path.replace(f"{os.sep}imgs{os.sep}", f"{os.sep}{mask_folder}{os.sep}")
    else:
        mask_path = os.path.join(os.path.dirname(sample.path), mask_folder, os.path.basename(sample.path))
    if not os.path.exists(mask_path):
        return None
    mask = tifffile.imread(mask_path)
    if mask.ndim > 2:
        mask = mask.squeeze()
    if crop_size is not None:
        mask = center_crop_mask(mask, crop_size)
    return mask


def flatten_channels(img: np.ndarray, channel_indices: Sequence[int]) -> np.ndarray:
    if not channel_indices:
        return np.empty((img.shape[1] * img.shape[2], 0), dtype=img.dtype)
    subset = img[channel_indices]
    return subset.reshape(len(channel_indices), -1).transpose(1, 0)


def compute_r2(features: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> List[float]:
    assigned = centers[labels]
    sse = ((features - assigned) ** 2).sum(axis=0)
    denom = ((features - features.mean(axis=0)) ** 2).sum(axis=0)
    denom = np.clip(denom, 1e-8, None)
    return (1.0 - sse / denom).tolist()


def plot_dataset_metrics(dataset: str, metrics: Dict, out_dir: str, target_r2: float, entity_name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # R2 per marker histogram
    plt.figure(figsize=(6, 4))
    r2_vals = metrics.get("r2_per_marker", [])
    if r2_vals:
        plt.hist(r2_vals, bins=30, color="#4c78a8", alpha=0.8)
    plt.axvline(target_r2, color="red", linestyle="--", label=f"target {target_r2:.2f}")
    plt.title(f"{dataset}: R^2 per marker")
    plt.xlabel("R^2")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{dataset}_r2_hist.png"), dpi=150)
    plt.close()

    # Cluster variance scatter (intersection vs non-intersection)
    inter_vars = [c["intersection_var_mean"] for c in metrics.get("cluster_stats", []) if c.get("intersection_var_mean") is not None]
    out_vars = [c["non_intersection_var_mean"] for c in metrics.get("cluster_stats", []) if c.get("non_intersection_var_mean") is not None]
    if inter_vars:
        plt.figure(figsize=(6, 4))
        plt.scatter(inter_vars, out_vars if out_vars else [0.0] * len(inter_vars), alpha=0.6, s=10, color="#f58518")
        plt.xlabel("Intersection variance (cluster mean)")
        plt.ylabel("Non-intersection variance" + (" (cluster mean)" if out_vars else " (none)") )
        plt.title(f"{dataset}: cluster variances ({entity_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{dataset}_variance_scatter.png"), dpi=150)
        plt.close()


def summarise(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "count": int(arr.size),
    }


def cluster_and_measure(
    intersection_feats: np.ndarray,
    out_feats: Optional[np.ndarray],
    n_clusters: int,
    random_state: int,
) -> Dict:
    n_clusters = min(n_clusters, max(1, intersection_feats.shape[0]))
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = model.fit_predict(intersection_feats)
    centers = model.cluster_centers_
    r2_per_marker = compute_r2(intersection_feats, labels, centers)

    cluster_stats = []
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        if not np.any(mask):
            continue
        inter_var = float(np.mean(np.var(intersection_feats[mask], axis=0)))
        out_var = None
        if out_feats is not None and out_feats.shape[1] > 0:
            out_var = float(np.mean(np.var(out_feats[mask], axis=0)))
        cluster_stats.append(
            {
                "cluster": int(cluster_id),
                "size": int(mask.sum()),
                "intersection_var_mean": inter_var,
                "non_intersection_var_mean": out_var,
            }
        )

    return {
        "n_clusters": n_clusters,
        "r2_per_marker": r2_per_marker,
        "r2_summary": summarise(r2_per_marker),
        "cluster_stats": cluster_stats,
        "intersection_var_summary": summarise([c["intersection_var_mean"] for c in cluster_stats]),
        "non_intersection_var_summary": summarise([
            c["non_intersection_var_mean"] for c in cluster_stats if c["non_intersection_var_mean"] is not None
        ]),
    }


def collect_pixels(
    samples: Iterable[Sample],
    intersection_idx: Sequence[int],
    out_idx: Sequence[int],
    max_pixels_per_image: Optional[int],
    rng: np.random.Generator,
    total_images: Optional[int],
    desc: str,
) -> Dict[str, np.ndarray]:
    inter_rows: List[np.ndarray] = []
    out_rows: List[np.ndarray] = []
    for sample in tqdm(samples, total=total_images, desc=desc, leave=False):
        inter = flatten_channels(sample.image, intersection_idx)
        out = flatten_channels(sample.image, out_idx) if out_idx else None
        if max_pixels_per_image is not None and inter.shape[0] > max_pixels_per_image:
            keep = rng.choice(inter.shape[0], size=max_pixels_per_image, replace=False)
            inter = inter[keep]
            if out is not None and out.shape[0] > 0:
                out = out[keep]
        inter_rows.append(inter)
        if out is not None:
            out_rows.append(out)
    return {
        "intersection": np.concatenate(inter_rows, axis=0) if inter_rows else np.empty((0, len(intersection_idx)), dtype=np.float32),
        "out": np.concatenate(out_rows, axis=0) if out_rows else None,
    }


def collect_cells(
    samples: Iterable[Sample],
    intersection_idx: Sequence[int],
    out_idx: Sequence[int],
    max_cells_per_image: Optional[int],
    rng: np.random.Generator,
    mask_folder: str,
    crop_size: Optional[int],
    total_images: Optional[int],
    desc: str,
) -> Dict[str, np.ndarray]:
    inter_rows: List[np.ndarray] = []
    out_rows: List[np.ndarray] = []
    for sample in tqdm(samples, total=total_images, desc=desc, leave=False):
        mask = load_mask_for_sample(sample, mask_folder=mask_folder, crop_size=crop_size)
        if mask is None:
            continue
        cell_ids = np.unique(mask)
        cell_ids = cell_ids[cell_ids != 0]
        if cell_ids.size == 0:
            continue
        image = sample.image
        inter_features = []
        out_features = [] if out_idx else None
        for cid in cell_ids:
            region = mask == cid
            if not np.any(region):
                continue
            inter_feat = image[intersection_idx][:, region].mean(axis=1) if intersection_idx else np.empty(0, dtype=image.dtype)
            inter_features.append(inter_feat)
            if out_idx:
                out_feat = image[out_idx][:, region].mean(axis=1)
                out_features.append(out_feat)
        if not inter_features:
            continue
        inter_feats = np.stack(inter_features, axis=0)
        out_feats = np.stack(out_features, axis=0) if out_idx else None
        if max_cells_per_image is not None and inter_feats.shape[0] > max_cells_per_image:
            keep = rng.choice(inter_feats.shape[0], size=max_cells_per_image, replace=False)
            inter_feats = inter_feats[keep]
            if out_feats is not None:
                out_feats = out_feats[keep]
        inter_rows.append(inter_feats)
        if out_feats is not None:
            out_rows.append(out_feats)
    return {
        "intersection": np.concatenate(inter_rows, axis=0) if inter_rows else np.empty((0, len(intersection_idx)), dtype=np.float32),
        "out": np.concatenate(out_rows, axis=0) if out_rows else None,
    }


def compute_marker_variance_baseline(
    panel_config_path: str,
    tokenizer_path: str,
    split: str = "test",
    crop_size: Optional[int] = None,
    datasets: Optional[Sequence[str]] = None,
    n_clusters: int = 1000,
    target_r2: float = 0.1,
    max_pixels: Optional[int] = 500_000,
    max_pixels_per_image: Optional[int] = 50_000,
    use_butterworth: bool = True,
    use_median: bool = False,
    use_minmax: bool = False,
    allow_missing: bool = False,
    seed: int = 0,
    evaluation_mode: str = "pixel",
    mask_folder: str = "masks",
    max_images_per_dataset: Optional[int] = None,
) -> Dict[str, Dict]:
    rng = np.random.default_rng(seed)
    panel_config = load_yaml(panel_config_path)
    tokenizer = load_yaml(tokenizer_path)
    datasets_to_use = datasets or panel_config.get("datasets", [])
    results: Dict[str, Dict] = {}

    for dataset_name in tqdm(datasets_to_use):
        if dataset_name == "nsclc2-panel1":
            continue
        marker_names = panel_config["markers"].get(dataset_name, [])
        intersection = [m for m in marker_names if m in PANEL_1_MARKER_NAMES]
        if not intersection:
            continue
        non_intersection = [m for m in marker_names if m not in intersection]
        intersection_idx = [idx for idx, name in enumerate(marker_names) if name in intersection]
        out_idx = [idx for idx, name in enumerate(marker_names) if name not in intersection]

        samples, n_images_used = build_dataset_generator(
            panel_config=panel_config,
            tokenizer=tokenizer,
            split=split,
            crop_size=crop_size,
            dataset_name=dataset_name,
            use_butterworth=use_butterworth,
            use_median=use_median,
            use_minmax=use_minmax,
            allow_missing=allow_missing,
            max_images=max_images_per_dataset,
            rng=rng,
        )
        if evaluation_mode == "pixel":
            pixels = collect_pixels(
                samples=samples,
                intersection_idx=intersection_idx,
                out_idx=out_idx,
                max_pixels_per_image=max_pixels_per_image,
                rng=rng,
                total_images=n_images_used,
                desc=f"{dataset_name} (pixels)",
            )
        elif evaluation_mode == "cell":
            pixels = collect_cells(
                samples=samples,
                intersection_idx=intersection_idx,
                out_idx=out_idx,
                max_cells_per_image=max_pixels_per_image,
                rng=rng,
                mask_folder=mask_folder,
                crop_size=crop_size,
                total_images=n_images_used,
                desc=f"{dataset_name} (cells)",
            )
        else:
            raise ValueError(f"Unsupported evaluation_mode: {evaluation_mode}")

        inter_feats = pixels["intersection"]
        out_feats = pixels["out"]
        if inter_feats.shape[0] == 0:
            continue
        if max_pixels is not None and inter_feats.shape[0] > max_pixels:
            keep = rng.choice(inter_feats.shape[0], size=max_pixels, replace=False)
            inter_feats = inter_feats[keep]
            if out_feats is not None and out_feats.shape[0] > 0:
                out_feats = out_feats[keep]
        metrics = cluster_and_measure(
            intersection_feats=inter_feats,
            out_feats=out_feats,
            n_clusters=n_clusters,
            random_state=seed,
        )
        metrics.update(
            {
                "dataset": dataset_name,
                "intersection_markers": intersection,
                "non_intersection_markers": non_intersection,
                "evaluation_mode": evaluation_mode,
                "n_entities_used": int(inter_feats.shape[0]),
                "n_images_used": int(n_images_used),
                "target_r2": target_r2,
                "hit_target_r2": metrics["r2_summary"]["mean"] <= target_r2,
            }
        )
        results[dataset_name] = metrics
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline variance analysis for virtual staining.")
    parser.add_argument("--config", required=True, help="Training config to read defaults (panel/tokenizer).")
    parser.add_argument("--panel-config", default=None, help="Override path to panel config YAML.")
    parser.add_argument("--tokenizer-config", default=None, help="Override path to tokenizer YAML.")
    parser.add_argument("--split", default="test", help="Dataset split to use.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional subset of datasets to process.")
    parser.add_argument("--crop-size", type=int, default=None, help="Optional center crop size.")
    parser.add_argument("--n-clusters", type=int, default=1000, help="KMeans cluster count (default: 1000).")
    parser.add_argument("--target-r2", type=float, default=0.1, help="Target mean R^2 threshold for intersection markers.")
    parser.add_argument("--max-pixels", type=int, default=500_000, help="Global pixel cap per dataset (after stacking).")
    parser.add_argument("--max-pixels-per-image", type=int, default=50_000, help="Per-image pixel cap before stacking.")
    parser.add_argument("--max-images-per-dataset", type=int, default=None, help="Optional cap on images sampled per dataset before aggregation.")
    parser.add_argument("--mode", choices=["pixel", "cell"], default="pixel", help="Evaluate per pixel or per cell (mask required for cells).")
    parser.add_argument("--mask-folder", default="masks", help="Folder name containing masks alongside imgs (default: masks).")
    parser.add_argument("--disable-butterworth", action="store_true", help="Skip Butterworth filtering.")
    parser.add_argument("--enable-median-denoising", action="store_true", help="Enable median filtering.")
    parser.add_argument("--use-minmax-normalization", action="store_true", help="Use min-max normalization instead of clipping.")
    parser.add_argument("--allow-missing-images", action="store_true", help="Skip datasets with no images instead of raising.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling and clustering.")
    parser.add_argument("--output", default="marker_variance_baseline.json", help="Path to save JSON results.")
    parser.add_argument("--figures-dir", default=None, help="Optional directory to save per-dataset plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    panel_config_path = args.panel_config or config.get("panel_config")
    tokenizer_path = args.tokenizer_config or config.get("tokenizer_config")
    if not panel_config_path or not tokenizer_path:
        raise ValueError("panel_config and tokenizer_config must be provided via config or CLI flags.")

    crop_size = infer_crop_size(config, args.crop_size)
    results = compute_marker_variance_baseline(
        panel_config_path=panel_config_path,
        tokenizer_path=tokenizer_path,
        split=args.split,
        datasets=args.datasets,
        crop_size=crop_size,
        n_clusters=args.n_clusters,
        target_r2=args.target_r2,
        max_pixels=args.max_pixels,
        max_pixels_per_image=args.max_pixels_per_image,
        use_butterworth=not args.disable_butterworth,
        use_median=args.enable_median_denoising,
        use_minmax=args.use_minmax_normalization,
        allow_missing=args.allow_missing_images,
        seed=args.seed,
        evaluation_mode=args.mode,
        mask_folder=args.mask_folder,
        max_images_per_dataset=args.max_images_per_dataset,
    )

    total_images = sum(v.get("n_images_used", 0) for v in results.values())
    total_entities = sum(v.get("n_entities_used", 0) for v in results.values())
    entity_name = "pixels" if args.mode == "pixel" else "cells"

    if args.figures_dir:
        os.makedirs(args.figures_dir, exist_ok=True)
        for ds_name, metrics in results.items():
            plot_dataset_metrics(ds_name, metrics, args.figures_dir, target_r2=args.target_r2, entity_name=entity_name)

    payload = {
        "config": {
            "config_path": os.path.abspath(args.config),
            "panel_config": os.path.abspath(panel_config_path),
            "tokenizer_config": os.path.abspath(tokenizer_path),
            "split": args.split,
            "crop_size": crop_size,
            "n_clusters": args.n_clusters,
            "target_r2": args.target_r2,
            "max_pixels": args.max_pixels,
            "max_pixels_per_image": args.max_pixels_per_image,
            "max_images_per_dataset": args.max_images_per_dataset,
            "mode": args.mode,
            "mask_folder": args.mask_folder,
            "use_butterworth": not args.disable_butterworth,
            "use_median": args.enable_median_denoising,
            "use_minmax": args.use_minmax_normalization,
        },
        "summary": {
            "total_datasets": len(results),
            "total_images_used": total_images,
            "total_entities_used": total_entities,
            "entity_name": entity_name,
        },
        "results": results,
    }

    with open(args.output, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved variance baseline metrics for {len(results)} dataset(s) to {args.output}")
    print(f"Total images used: {total_images}, total {entity_name}: {total_entities}")


if __name__ == "__main__":
    main()
