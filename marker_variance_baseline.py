"""Baseline clustering-based variance analysis for inpainted markers.

This script scans all datasets defined in a panel config (except nsclc2-panel1),
clusters pixels or cells using the intersection of Immucan Panel 1 markers, and
reports within-cluster variances for both the intersecting markers and the
remaining markers. It is meant as a baseline for virtual staining performance.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile
from ruamel.yaml import YAML
from sklearn.cluster import KMeans
from tqdm import tqdm

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
        use_median_denoising=False,
        use_butterworth_filter=False,
        use_minmax_normalization=False,
        use_clip_normalization=True,
        use_preprocessing=False,
        file_extension="npy",
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


def plot_non_intersection_variances(dataset: str, metrics: Dict, out_dir: str, max_markers: int = 30) -> None:
    var_map: Dict[str, List[float]] = metrics.get("non_intersection_var_per_marker", {}) or {}
    r2_map: Dict[str, List[float]] = metrics.get("non_intersection_r2_per_marker", {}) or {}
    global_var_map: Dict[str, float] = metrics.get("non_intersection_global_var_per_marker", {}) or {}
    cluster_stats: List[Dict] = metrics.get("cluster_stats", []) or []
    cluster_sizes = np.array([c.get("size", 0) for c in cluster_stats], dtype=float)

    if not var_map and not r2_map:
        return
    # Limit to most variable markers to keep legend readable
    marker_means = [(name, float(np.mean(vals)) if len(vals) > 0 else 0.0) for name, vals in var_map.items()]
    marker_means.sort(key=lambda x: x[1], reverse=True)
    selected = [name for name, _ in marker_means[:max_markers]] if marker_means else list(r2_map.keys())[:max_markers]

    if var_map:
        plt.figure(figsize=(7, 5))
        for name in selected:
            vals = var_map.get(name, [])
            if not vals:
                continue
            plt.hist(vals, bins=25, alpha=0.4, label=name)
        plt.title(f"{dataset}: variance hist (markers outside intersection)")
        plt.xlabel("Variance across clusters")
        plt.ylabel("Count")
        plt.legend(fontsize="small", ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{dataset}_non_intersection_hist.png"), dpi=150)
        plt.close()

        # Bar plot of mean variance per marker (selected)
        means = [np.mean(var_map[name]) if var_map.get(name) else 0.0 for name in selected]
        plt.figure(figsize=(max(6, 0.5 * len(selected)), 4))
        plt.bar(selected, means, color="#4c78a8")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Mean variance across clusters")
        plt.title(f"{dataset}: mean variance per marker (outside intersection)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{dataset}_non_intersection_bar.png"), dpi=150)
        plt.close()

        # Boxplots for variance per marker
        data = [var_map[name] for name in selected if var_map.get(name)]
        labels = [name for name in selected if var_map.get(name)]
        if data:
            plt.figure(figsize=(max(6, 0.6 * len(labels)), 4))
            plt.boxplot(data, labels=labels, showfliers=False)
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Variance across clusters")
            plt.title(f"{dataset}: variance boxplots (outside intersection)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{dataset}_non_intersection_var_box.png"), dpi=150)
            plt.close()

 # --- 2) NOWY: histogramy relatywnej wariancji (Var_cluster / Var_global) ---
        # per-marker, overlayed just like raw variance
    if global_var_map:
        plt.figure(figsize=(7, 5))
        for name in selected:
            vals = var_map.get(name, [])
            gvar = global_var_map.get(name)
            if not vals or gvar is None or gvar <= 1e-8:
                continue
            rel_vals = [v / gvar for v in vals]
            plt.hist(rel_vals, bins=25, alpha=0.4, label=name)
        plt.title(f"{dataset}: relative variance hist (markers outside intersection)")
        plt.xlabel("Relative variance (Var_cluster / Var_global)")
        plt.ylabel("Count")
        plt.legend(fontsize="small", ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{dataset}_non_intersection_relvar_hist.png"), dpi=150)
        plt.close()

    # Boxplots for R^2 per marker (how well clusters explain each marker)
    if r2_map:
        r2_selected = [name for name in selected if name in r2_map and r2_map[name]]
        if not r2_selected:
            r2_selected = [name for name, vals in r2_map.items() if vals][:max_markers]
        data = [r2_map[name] for name in r2_selected if r2_map.get(name)]
        labels = [name for name in r2_selected if r2_map.get(name)]
        if data:
            plt.figure(figsize=(max(6, 0.6 * len(labels)), 4))
            plt.boxplot(data, labels=labels, showfliers=False)
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Cluster R^2")
            plt.title(f"{dataset}: R^2 boxplots (outside intersection)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{dataset}_non_intersection_r2_box.png"), dpi=150)
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
    out_marker_names: Optional[Sequence[str]] = None,
    debug_logging: bool = False,
) -> Dict:
    n_clusters = min(n_clusters, max(1, intersection_feats.shape[0]))
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = model.fit_predict(intersection_feats)
    centers = model.cluster_centers_
    r2_per_marker = compute_r2(intersection_feats, labels, centers) # computes R2 (how much of the global variance is explained by clustering) for each marker in the intersection
    if debug_logging:
        print(
            "[cluster_and_measure] intersection_feats:",
            intersection_feats.shape,
            "out_feats:",
            None if out_feats is None else out_feats.shape,
            "labels:",
            labels.shape,
            "centers:",
            centers.shape,
            "r2_per_marker_len:",
            len(r2_per_marker),
        )

    cluster_stats = []
    cluster_ids_used: List[int] = []
    out_marker_var: Dict[str, List[float]] = {name: [] for name in (out_marker_names or [])}
    out_marker_mean: Dict[str, List[float]] = {name: [] for name in (out_marker_names or [])}
    out_marker_r2: Dict[str, List[float]] = {name: [] for name in (out_marker_names or [])}
    # Precompute global denom for R2 per marker (outside intersection)
    global_out_denom = None
    global_out_var = None
    if out_feats is not None and out_feats.shape[1] > 0:
        global_out_denom = ((out_feats - out_feats.mean(axis=0)) ** 2).sum(axis=0)
        global_out_denom = np.clip(global_out_denom, 1e-8, None)
        global_out_var = np.var(out_feats, axis=0)
        # już masz global_out_denom, ale tu chcemy zwykłą wariancję, nie sumę kwadratów

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        if not np.any(mask):
            continue
        cluster_ids_used.append(cluster_id)
        inter_var = float(np.mean(np.var(intersection_feats[mask], axis=0)))
        out_var = None
        if out_feats is not None and out_feats.shape[1] > 0:
            out_var = float(np.mean(np.var(out_feats[mask], axis=0)))
            if out_marker_names:
                var_per_marker = np.var(out_feats[mask], axis=0)
                mean_per_marker = np.mean(out_feats[mask], axis=0)
                for idx, name in enumerate(out_marker_names):
                    if idx < var_per_marker.shape[0]:
                        out_marker_var[name].append(float(var_per_marker[idx]))
                        out_marker_mean[name].append(float(mean_per_marker[idx]))
                if global_out_denom is not None:
                    sse_cluster = ((out_feats[mask] - out_feats[mask].mean(axis=0)) ** 2).sum(axis=0)
                    r2_cluster = 1.0 - sse_cluster / global_out_denom
                    for idx, name in enumerate(out_marker_names):
                        if idx < r2_cluster.shape[0]:
                            out_marker_r2[name].append(float(r2_cluster[idx]))
        cluster_stats.append(
            {
                "cluster": int(cluster_id),
                "size": int(mask.sum()),
                "intersection_var_mean": inter_var,
                "non_intersection_var_mean": out_var,
            }
        )
        if debug_logging and cluster_id < 3:  # log a few clusters to avoid spamming
            print(
                f"[cluster_and_measure] cluster {cluster_id}: size={int(mask.sum())}, "
                f"intersection_var_mean={inter_var:.4f}, "
                f"non_intersection_var_mean={out_var if out_var is None else round(out_var, 4)}"
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
        "non_intersection_var_per_marker": out_marker_var if out_marker_var else {},
        "non_intersection_mean_per_marker": out_marker_mean if out_marker_mean else {},
        "non_intersection_r2_per_marker": out_marker_r2 if out_marker_r2 else {},
        "non_intersection_global_var_per_marker": (
            {name: float(global_out_var[idx]) for idx, name in enumerate(out_marker_names)}
            if (out_marker_names is not None and global_out_var is not None)
            else {}
        ),

        "cluster_ids": cluster_ids_used,
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
) -> Tuple[Dict[str, np.ndarray], int, int]:
    inter_rows: List[np.ndarray] = []
    out_rows: List[np.ndarray] = []
    images_with_masks = 0
    total_cells = 0
    for sample in tqdm(samples, total=total_images, desc=desc, leave=False):
        mask = load_mask_for_sample(sample, mask_folder=mask_folder, crop_size=crop_size)
        if mask is None:
            continue
        images_with_masks += 1
        cell_ids = np.unique(mask)
        cell_ids = cell_ids[cell_ids != 0]
        if cell_ids.size == 0:
            continue
        total_cells += int(cell_ids.size)
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
    }, images_with_masks, total_cells


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
    debug_logging: bool = False,
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
            print(f"Dataset {dataset_name}: collected {pixels['intersection'].shape} pixels from {n_images_used} images.")
            print(f"outside intersection shape: {pixels['out'].shape if pixels['out'] is not None else None}")
            images_with_masks = n_images_used
            entities_count = pixels["intersection"].shape[0]
        elif evaluation_mode == "cell":
            pixels, images_with_masks, entities_count = collect_cells(
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
            out_marker_names=non_intersection,
            debug_logging=debug_logging,
        )
        metrics.update(
            {
                "dataset": dataset_name,
                "intersection_markers": intersection,
                "non_intersection_markers": non_intersection,
                "evaluation_mode": evaluation_mode,
                "n_entities_used": int(inter_feats.shape[0]),
                "n_images_used": int(n_images_used),
                "n_images_with_masks": int(images_with_masks),
                "n_cells_considered": int(entities_count) if evaluation_mode == "cell" else None,
                "target_r2": target_r2,
                "hit_target_r2": metrics["r2_summary"]["mean"] <= target_r2,
            }
        )
        results[dataset_name] = metrics
    return results


def build_marker_stats_rows(results: Dict[str, Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for dataset, metrics in results.items():
        var_map: Dict[str, List[float]] = metrics.get("non_intersection_var_per_marker", {}) or {}
        mean_map: Dict[str, List[float]] = metrics.get("non_intersection_mean_per_marker", {}) or {}
        r2_map: Dict[str, List[float]] = metrics.get("non_intersection_r2_per_marker", {}) or {}
        global_var_map: Dict[str, float] = metrics.get("non_intersection_global_var_per_marker", {}) or {}
        cluster_ids: List[int] = metrics.get("cluster_ids", []) or []
        cluster_sizes = {c.get("cluster"): c.get("size") for c in metrics.get("cluster_stats", []) or []}
        eval_mode = metrics.get("evaluation_mode")
        n_images_used = metrics.get("n_images_used")
        n_entities_used = metrics.get("n_entities_used")
        for marker, var_list in var_map.items():
            mean_list = mean_map.get(marker, []) or []
            r2_list = r2_map.get(marker, []) or []
            for idx, var in enumerate(var_list):
                cid = cluster_ids[idx] if idx < len(cluster_ids) else idx
                rows.append(
                    {
                        "dataset": dataset,
                        "marker": marker,
                        "cluster_id": cid,
                        "cluster_size": cluster_sizes.get(cid),
                        "marker_variance": var,
                        "marker_mean": mean_list[idx] if idx < len(mean_list) else None,
                        "marker_r2": r2_list[idx] if idx < len(r2_list) else None,
                        "marker_global_var": global_var_map.get(marker),
                        "evaluation_mode": eval_mode,
                        "n_images_used": n_images_used,
                        "n_entities_used": n_entities_used,
                    }
                )
    # Sort rows by marker mean variance across datasets
    if not rows:
        return rows
    df = pd.DataFrame(rows)
    marker_order = (
        df.groupby("marker")["marker_variance"].mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    df["marker"] = pd.Categorical(df["marker"], categories=marker_order, ordered=True)
    df = df.sort_values(["marker", "dataset", "cluster_id"], na_position="last")
    return df.to_dict(orient="records")


def save_marker_dataset_heatmap(df: pd.DataFrame, figures_dir: str,
                                top_n: int = 1000) -> None:
    """
    Heatmapa średniej wariancji markera w każdym datasecie.
    Oś Y: markery (top_n), oś X: datasety.
    """
    if df.empty:
        return

    agg = df.groupby(["marker", "dataset"])["marker_variance"].mean().reset_index()

    # wyznacz kolejność markerów jak wcześniej (top_n najbardziej zmiennych)
    marker_order = (
        agg.groupby("marker")["marker_variance"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    top_markers = marker_order[:top_n]

    pivot = agg[agg["marker"].isin(top_markers)].pivot(
        index="marker", columns="dataset", values="marker_variance"
    )

    if pivot.empty:
        return

    plt.figure(figsize=(max(10, 0.5 * pivot.shape[1]), 0.5 * pivot.shape[0] + 3))
    sns.heatmap(
        pivot,
        cmap="viridis",
        linewidths=0.5,
        linecolor="grey",
        cbar_kws={"label": "Mean variance across clusters"},
    )
    plt.ylabel("Marker (outside intersection)")
    plt.xlabel("Dataset")
    plt.title("Non-intersection markers: variance per dataset")
    plt.tight_layout()

    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, "marker_dataset_variance_heatmap.png"), dpi=200)
    plt.close()

def save_marker_variance_vs_r2_scatter(df: pd.DataFrame, figures_dir: str,
                                       min_points: int = 5) -> None:
    """
    Dla każdego markera liczymy:
    - mean marker_variance
    - median marker_r2
    i rysujemy scatter: x = mean variance, y = median R^2.

    Dzięki temu można zobaczyć markery:
    - o wysokiej wariancji i niskim R² (klastry 'nie łapią' markera),
    - o niskiej wariancji i wysokim R² (dobrze zachowujące się markery).
    """
    if df.empty or "marker_r2" not in df.columns:
        return

    # pozbywamy się NaN
    sub = df.dropna(subset=["marker_variance", "marker_r2"])

    marker_stats = (
        sub.groupby("marker")
        .agg(
            mean_var=("marker_variance", "mean"),
            median_r2=("marker_r2", "median"),
            count=("marker_variance", "size"),
        )
        .reset_index()
    )

    marker_stats = marker_stats[marker_stats["count"] >= min_points]
    if marker_stats.empty:
        return

    plt.figure(figsize=(7, 6))
    plt.scatter(marker_stats["mean_var"], marker_stats["median_r2"], alpha=0.7)
    for _, row in marker_stats.iterrows():
        # etykieta tylko dla najbardziej ekstremalnych markerów
        if row["mean_var"] > marker_stats["mean_var"].quantile(0.9) or \
           row["median_r2"] < marker_stats["median_r2"].quantile(0.1):
            plt.text(row["mean_var"], row["median_r2"], row["marker"],
                     fontsize=7, ha="left", va="bottom")

    plt.xlabel("Mean variance across clusters")
    plt.ylabel("Median R² (how well clusters explain the marker)")
    plt.title("Markers: variance vs. cluster R²")
    plt.tight_layout()

    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, "markers_variance_vs_r2_scatter.png"), dpi=200)
    plt.close()


# def save_combined_marker_barplot(df: pd.DataFrame, figures_dir: str, top_n: int = 1000,
#                                  min_datasets: int = 1) -> None:
#     """
#     Zbiorczy wykres: dla każdego markera pokazujemy średnią wariancję
#     (uśrednioną po datasetach) + odchylenie standardowe jako errorbar.

#     Dzięki temu:
#     - nie ma gigantycznej legendy z nazwami datasetów,
#     - oś Y to markery (łatwiej czytać długie nazwy),
#     - widzimy zarówno poziom wariancji, jak i jej zmienność między datasetami.
#     """
#     if df.empty:
#         return

#     # Średnia wariancja per (marker, dataset)
#     agg = df.groupby(["marker", "dataset"])["marker_variance"].mean().reset_index()

#     # Agregacja po markerach: średnia, std i liczba datasetów
#     marker_stats = (
#         agg.groupby("marker")["marker_variance"]
#         .agg(["mean", "std", "count"])
#         .reset_index()
#         .rename(columns={"mean": "mean_var", "std": "std_var", "count": "n_datasets"})
#     )

#     # Opcjonalnie odfiltruj markery, które pojawiają się tylko w pojedynczych datasetach
#     marker_stats = marker_stats[marker_stats["n_datasets"] >= min_datasets]
#     if marker_stats.empty:
#         return

#     # Wybrać top_n markerów o największej średniej wariancji
#     marker_stats = marker_stats.sort_values("mean_var", ascending=False)
#     top_markers = marker_stats.head(top_n)

#     # Rysujemy poziome słupki
#     plt.figure(figsize=(max(8, 0.35 * len(top_markers)), 6))
#     y_pos = np.arange(len(top_markers))

#     plt.barh(
#         y_pos,
#         top_markers["mean_var"],
#         xerr=top_markers["std_var"].fillna(0.0),
#         alpha=0.8,
#         capsize=3,
#     )
#     plt.yticks(y_pos, top_markers["marker"])
#     plt.gca().invert_yaxis()  # największa wariancja na górze
#     plt.xlabel("Mean variance across clusters (averaged over datasets)")
#     plt.title("Non-intersection markers: mean variance ± std across datasets")
#     plt.tight_layout()

#     os.makedirs(figures_dir, exist_ok=True)
#     plt.savefig(os.path.join(figures_dir, "all_markers_variance_bar.png"), dpi=200)
#     plt.close()


# def save_combined_relative_variance_barplot(df: pd.DataFrame, figures_dir: str,
#                                             top_n: int = 1000) -> None:
#     """
#     Wykres: dla każdego markera pokazujemy relatywną wariancję:
#         rel_var = mean(within_cluster_var / global_var)
#     czyli 'jaki ułamek wariancji markera pozostaje w klastrach'.

#     Dzięki temu możemy porównywać markery między sobą – metryka jest bezjednostkowa.
#     """
#     if df.empty:
#         return

#     # potrzebujemy tylko wierszy, gdzie znamy globalną wariancję
#     sub = df.dropna(subset=["marker_variance", "marker_global_var"]).copy()
#     # zabezpieczenie przed 0 w mianowniku
#     sub = sub[sub["marker_global_var"] > 1e-8]
#     if sub.empty:
#         return

#     sub["rel_var"] = sub["marker_variance"] / sub["marker_global_var"]

#     # uśredniamy rel_var po datasetach (możesz też zrobić osobne wykresy per dataset)
#     agg = sub.groupby("marker")["rel_var"].mean().reset_index()

#     # wybieramy top_n markerów o największej relatywnej wariancji
#     agg = agg.sort_values("rel_var", ascending=False)
#     top = agg.head(top_n)

#     plt.figure(figsize=(max(8, 0.4 * len(top)), 6))
#     plt.bar(top["marker"], top["rel_var"])
#     plt.xticks(rotation=60, ha="right")
#     plt.ylabel("Relative within-cluster variance (Var_within / Var_global)")
#     plt.xlabel("Marker (outside intersection)")
#     plt.title("Non-intersection markers: relative variance across clusters")
#     plt.tight_layout()

#     os.makedirs(figures_dir, exist_ok=True)
#     plt.savefig(os.path.join(figures_dir, "all_markers_relative_variance_bar.png"), dpi=200)
#     plt.close()

def save_combined_marker_barplot(
    df: pd.DataFrame,
    figures_dir: str,
    top_n: int = 1000,
    min_datasets: int = 1,
) -> None:
    """
    1) Grouped bar plot: dla każdego markera pokazujemy średnią wariancję
       w każdym datasecie osobno (kolor = dataset).
    2) Dodatkowy wykres: uśrednienie po datasetach + std jako errorbar
       (jeden słupek na marker).

    - Oś X (grouped): markery (outside intersection)
    - Oś Y: mean(marker_variance)
    - Kolor: dataset
    """
    if df.empty:
        return

    # Średnia wariancja per (marker, dataset)
    agg = df.groupby(["marker", "dataset"])["marker_variance"].mean().reset_index()
    if agg.empty:
        return

    # W ilu datasetach pojawia się dany marker
    counts = (
        agg.groupby("marker")["dataset"]
        .nunique()
        .reset_index(name="n_datasets")
    )
    agg = agg.merge(counts, on="marker", how="left")

    # Odfiltruj markery zbyt rzadkie
    agg = agg[agg["n_datasets"] >= min_datasets]
    if agg.empty:
        return

    # Ranking markerów po globalnej średniej wariancji (uśrednionej po datasetach)
    marker_order = (
        agg.groupby("marker")["marker_variance"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    top_markers = marker_order[:top_n]

    plot_df = agg[agg["marker"].isin(top_markers)]
    if plot_df.empty:
        return

    # --- WYKRES 1: GROUPED BAR PLOT (marker × dataset) --------------------
    pivot = plot_df.pivot(index="marker", columns="dataset", values="marker_variance")
    pivot = pivot.loc[top_markers]  # zachowaj kolejność

    markers = list(pivot.index)
    datasets = list(pivot.columns)
    n_markers = len(markers)
    n_datasets = len(datasets)

    if n_markers == 0 or n_datasets == 0:
        return

    x = np.arange(n_markers)

    bar_width_factor = 1.6
    bar_width = min(bar_width_factor * 0.8 / n_datasets, 0.35)

    plt.figure(figsize=(max(10, 0.6 * n_markers), 6))

    for i, ds in enumerate(datasets):
        vals = pivot[ds].values
        offset = (i - (n_datasets - 1) / 2) * bar_width

        bars = plt.bar(x + offset, vals, width=bar_width, label=ds)

        # --- NUMERKI NAD SŁUPKAMI ---
        for bar, val in zip(bars, vals):
            if np.isnan(val):
                continue
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    plt.xticks(x, markers, rotation=60, ha="right")
    plt.ylabel("Mean variance across clusters")
    plt.xlabel("Marker (outside intersection)")
    plt.title("Non-intersection markers: mean variance per dataset")
    plt.legend(title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, "all_markers_variance_bar.png"), dpi=200)
    plt.close()

    # --- WYKRES 2: MEAN ± STD PO DATASETACH (JEDEN SŁUPEK NA MARKER) ------
    # Bierzemy tylko top_markers
    agg_top = agg[agg["marker"].isin(top_markers)]

    marker_stats = (
        agg_top.groupby("marker")["marker_variance"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_var", "std": "std_var"})
    )

    if marker_stats.empty:
        return

    # Zachowaj kolejność jak w rankingu
    marker_stats["marker"] = pd.Categorical(
        marker_stats["marker"], categories=top_markers, ordered=True
    )
    marker_stats = marker_stats.sort_values("marker")

    y_pos = np.arange(len(marker_stats))

    plt.figure(figsize=(10, max(6, 0.4 * len(marker_stats))))

    means = marker_stats["mean_var"].values
    stds = marker_stats["std_var"].fillna(0.0).values

    bars = plt.barh(
        y_pos,
        means,
        xerr=stds,
        alpha=0.85,
        capsize=3,
    )

    plt.yticks(y_pos, marker_stats["marker"])
    plt.gca().invert_yaxis()
    plt.xlabel("Mean variance across clusters")
    plt.title("Non-intersection markers: mean variance ± std across datasets")

    # Numery przy słupkach
    for bar, m in zip(bars, means):
        x_right = bar.get_width()
        y_center = bar.get_y() + bar.get_height() / 2.0
        plt.text(
            x_right,
            y_center,
            f"{m:.2f}",
            ha="left",
            va="center",
            fontsize=7,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "all_markers_variance_bar_mean_std.png"), dpi=200)
    plt.close()

def save_combined_relative_variance_barplot(
    df: pd.DataFrame,
    figures_dir: str,
    top_n: int = 1000,
    min_datasets: int = 1,
) -> None:
    """
    1) Grouped bar plot relatywnej wariancji:
         rel_var = Var_within / Var_global
       per (marker, dataset), gdzie:
         Var_within = średnia ważona wariancji w klastrach (waga = cluster_size).
    2) Dodatkowy wykres: uśrednienie rel_var po datasetach + std jako errorbar.

    - Oś X (grouped): marker
    - Oś Y: rel_var (ułamek niewyjaśnionej wariancji)
    - Kolor: dataset
    """
    if df.empty:
        return

    sub = df.dropna(
        subset=["marker_variance", "marker_global_var", "cluster_size", "dataset"]
    ).copy()
    sub = sub[(sub["marker_global_var"] > 1e-8) & (sub["cluster_size"] > 0)]
    if sub.empty:
        return

    # Var_within per (marker, dataset) = średnia ważona Var_cluster wagami = cluster_size
    def _agg_group(g: pd.DataFrame) -> pd.Series:
        var_within = np.average(g["marker_variance"], weights=g["cluster_size"])
        global_var = g["marker_global_var"].iloc[0]
        rel_var = var_within / max(global_var, 1e-8)
        return pd.Series(
            {
                "var_within": var_within,
                "marker_global_var": global_var,
                "rel_var": rel_var,
            }
        )

    agg = (
        sub.groupby(["marker", "dataset"], as_index=False)
        .apply(_agg_group)
        .reset_index(drop=True)
    )
    if agg.empty:
        return

    # W ilu datasetach pojawia się dany marker
    counts = (
        agg.groupby("marker")["dataset"]
        .nunique()
        .reset_index(name="n_datasets")
    )
    agg = agg.merge(counts, on="marker", how="left")
    agg = agg[agg["n_datasets"] >= min_datasets]
    if agg.empty:
        return

    # Ranking markerów po globalnej średniej rel_var (uśrednionej po datasetach)
    marker_order = (
        agg.groupby("marker")["rel_var"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    top_markers = marker_order[:top_n]

    plot_df = agg[agg["marker"].isin(top_markers)]
    if plot_df.empty:
        return

    # --- WYKRES 1: GROUPED BAR PLOT (marker × dataset) --------------------
    pivot = plot_df.pivot(index="marker", columns="dataset", values="rel_var")
    pivot = pivot.loc[top_markers]

    markers = list(pivot.index)
    datasets = list(pivot.columns)
    n_markers = len(markers)
    n_datasets = len(datasets)

    if n_markers == 0 or n_datasets == 0:
        return

    x = np.arange(n_markers)
    bar_width_factor = 1.6
    bar_width = min(bar_width_factor * 0.8 / n_datasets, 0.35)

    plt.figure(figsize=(max(10, 0.6 * n_markers), 6))

    for i, ds in enumerate(datasets):
        vals = pivot[ds].values
        offset = (i - (n_datasets - 1) / 2) * bar_width

        bars = plt.bar(x + offset, vals, width=bar_width, label=ds)

        # --- NUMERKI NAD SŁUPKAMI ---
        for bar, val in zip(bars, vals):
            if np.isnan(val):
                continue
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    plt.xticks(x, markers, rotation=60, ha="right")
    plt.ylabel("Relative within-cluster variance (Var_within / Var_global)")
    plt.xlabel("Marker (outside intersection)")
    plt.title("Non-intersection markers: relative variance per dataset")
    plt.legend(title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, "all_markers_relative_variance_bar.png"), dpi=200)
    plt.close()

    # --- WYKRES 2: MEAN ± STD PO DATASETACH (JEDEN SŁUPEK NA MARKER) ------
    agg_top = agg[agg["marker"].isin(top_markers)]

    marker_stats = (
        agg_top.groupby("marker")["rel_var"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_rel_var", "std": "std_rel_var"})
    )
    if marker_stats.empty:
        return

    marker_stats["marker"] = pd.Categorical(
        marker_stats["marker"], categories=top_markers, ordered=True
    )
    marker_stats = marker_stats.sort_values("marker")

    y_pos = np.arange(len(marker_stats))
    plt.figure(figsize=(10, max(6, 0.4 * len(marker_stats))))

    means = marker_stats["mean_rel_var"].values
    stds = marker_stats["std_rel_var"].fillna(0.0).values

    bars = plt.barh(
        y_pos,
        means,
        xerr=stds,
        alpha=0.85,
        capsize=3,
    )

    plt.yticks(y_pos, marker_stats["marker"])
    plt.gca().invert_yaxis()
    plt.xlabel("Relative within-cluster variance (Var_within / Var_global)")
    plt.title("Non-intersection markers: relative variance ± std across datasets")

    for bar, m in zip(bars, means):
        x_right = bar.get_width()
        y_center = bar.get_y() + bar.get_height() / 2.0
        plt.text(
            x_right,
            y_center,
            f"{m:.2f}",
            ha="left",
            va="center",
            fontsize=7,
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(figures_dir, "all_markers_relative_variance_bar_mean_std.png"),
        dpi=200,
    )
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline variance analysis for virtual staining.")
    parser.add_argument("--config", required=False, help="Training config to read defaults (panel/tokenizer).")
    parser.add_argument("--panel-config", default=None, help="Override path to panel config YAML.")
    parser.add_argument("--tokenizer-config", default=None, help="Override path to tokenizer YAML.")
    parser.add_argument("--split", default="test", help="Dataset split to use.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional subset of datasets to process.")
    parser.add_argument("--crop-size", type=int, default=None, help="Optional center crop size.")
    parser.add_argument("--no-crop", action="store_true", help="Do not apply center crop (overrides crop-size/config).")
    parser.add_argument("--n-clusters", type=int, default=1000, help="KMeans cluster count (default: 1000).")
    parser.add_argument("--target-r2", type=float, default=0.1, help="Target mean R^2 threshold for intersection markers.")
    parser.add_argument("--max-pixels", type=int, default=None, help="Global pixel cap per dataset (after stacking).")
    parser.add_argument("--max-pixels-per-image", type=int, default=None, help="Per-image pixel cap before stacking.")
    parser.add_argument("--max-images-per-dataset", type=int, default=None, help="Optional cap on images sampled per dataset before aggregation.")
    parser.add_argument("--mode", choices=["pixel", "cell"], default="pixel", help="Evaluate per pixel or per cell (mask required for cells).")
    parser.add_argument("--mask-folder", default="masks", help="Folder name containing masks alongside imgs (default: masks).")
    parser.add_argument("--stats-csv", default=None, help="Optional path to save per-marker per-cluster stats (outside markers).")
    parser.add_argument("--debug-cluster-logging", action="store_true", help="Print shapes/cluster stats for debugging.")
    parser.add_argument("--disable-butterworth", action="store_true", help="Skip Butterworth filtering.")
    parser.add_argument("--enable-median-denoising", action="store_true", help="Enable median filtering.")
    parser.add_argument("--use-minmax-normalization", action="store_true", help="Use min-max normalization instead of clipping.")
    parser.add_argument("--allow-missing-images", action="store_true", help="Skip datasets with no images instead of raising.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling and clustering.")
    parser.add_argument("--output", default="marker_variance_baseline.json", help="Path to save JSON results.")
    parser.add_argument("--figures-dir", default=None, help="Optional directory to save per-dataset plots.")
    parser.add_argument("--results-path", default=None, help="If provided, load results JSON and regenerate plots/stats.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.results_path:
        with open(args.results_path, "r") as handle:
            payload = json.load(handle)
        results = payload.get("results", {})
        config = payload.get("config", {})
        crop_size = None if args.no_crop else infer_crop_size(config, args.crop_size)
        panel_config_path = config.get("panel_config")
        tokenizer_path = config.get("tokenizer_config")
        n_clusters = config.get("n_clusters", args.n_clusters)
        target_r2 = config.get("target_r2", args.target_r2)
        use_butterworth = config.get("use_butterworth", not args.disable_butterworth)
        use_median = config.get("use_median", args.enable_median_denoising)
        use_minmax = config.get("use_minmax", args.use_minmax_normalization)
        mode = config.get("mode", args.mode)
        max_images_per_dataset = config.get("max_images_per_dataset", args.max_images_per_dataset)
        max_pixels = config.get("max_pixels", args.max_pixels)
        max_pixels_per_image = config.get("max_pixels_per_image", args.max_pixels_per_image)
        split = config.get("split", args.split)
        mask_folder = config.get("mask_folder", args.mask_folder)
    else:
        if not args.config:
            raise ValueError("Either --config or --results-path must be provided.")
        config = load_yaml(args.config)
        panel_config_path = args.panel_config or config.get("panel_config")
        tokenizer_path = args.tokenizer_config or config.get("tokenizer_config")
        if not panel_config_path or not tokenizer_path:
            raise ValueError("panel_config and tokenizer_config must be provided via config or CLI flags.")

        crop_size = None if args.no_crop else infer_crop_size(config, args.crop_size)


        print("Computing marker variance baseline with the following settings:"
              f"\n  panel_config: {panel_config_path}"
              f"\n  tokenizer_config: {tokenizer_path}"
              f"\n  split: {args.split}"
              f"\n  datasets: {args.datasets or 'all in panel config'}"
              f"\n  crop_size: {crop_size}"
              f"\n  n_clusters: {args.n_clusters}"
              f"\n  target_r2: {args.target_r2}"
              f"\n  max_pixels: {args.max_pixels}"
              f"\n  max_pixels_per_image: {args.max_pixels_per_image}"
              f"\n  max_images_per_dataset: {args.max_images_per_dataset}"
              f"\n  evaluation_mode: {args.mode}"
              f"\n  use_butterworth: {not args.disable_butterworth}"
              f"\n  use_median_denoising: {args.enable_median_denoising}"
              f"\n  use_minmax_normalization: {args.use_minmax_normalization}"
              f"\n  allow_missing_images: {args.allow_missing_images}"
              f"\n  seed: {args.seed}"
              f"\n  mask_folder: {args.mask_folder}"
              f"\n  max_images_per_dataset: {args.max_images_per_dataset}")

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
            debug_logging=args.debug_cluster_logging,
        )
        n_clusters = args.n_clusters
        target_r2 = args.target_r2
        use_butterworth = not args.disable_butterworth
        use_median = args.enable_median_denoising
        use_minmax = args.use_minmax_normalization
        mode = args.mode
        max_images_per_dataset = args.max_images_per_dataset
        max_pixels = args.max_pixels
        max_pixels_per_image = args.max_pixels_per_image
        split = args.split
        mask_folder = args.mask_folder

    total_images = sum(v.get("n_images_used", 0) for v in results.values())
    total_entities = sum(v.get("n_entities_used", 0) for v in results.values())
    entity_name = "pixels" if mode == "pixel" else "cells"

    if args.figures_dir:
        os.makedirs(args.figures_dir, exist_ok=True)
        for ds_name, metrics in results.items():
            plot_dataset_metrics(ds_name, metrics, args.figures_dir, target_r2=target_r2, entity_name=entity_name)
            plot_non_intersection_variances(ds_name, metrics, args.figures_dir)

    # Build per-marker stats dataframe (outside markers)
    stats_rows = build_marker_stats_rows(results)
    if args.stats_csv:
        pd.DataFrame(stats_rows).to_csv(args.stats_csv, index=False)
        print(f"Saved marker stats to {args.stats_csv}")

    # Combined bar plot across datasets for outside markers
    if args.figures_dir and stats_rows:
        df_stats = pd.DataFrame(stats_rows)
        save_combined_marker_barplot(df_stats, args.figures_dir)
        save_marker_dataset_heatmap(df_stats, args.figures_dir)
        save_marker_variance_vs_r2_scatter(df_stats, args.figures_dir)
        save_combined_relative_variance_barplot(df_stats, args.figures_dir)

    payload = {
        "config": {
            "config_path": os.path.abspath(args.config) if args.config else None,
            "panel_config": os.path.abspath(panel_config_path) if panel_config_path else None,
            "tokenizer_config": os.path.abspath(tokenizer_path) if tokenizer_path else None,
            "split": split,
            "crop_size": crop_size,
            "n_clusters": n_clusters,
            "target_r2": target_r2,
            "max_pixels": max_pixels,
            "max_pixels_per_image": max_pixels_per_image,
            "max_images_per_dataset": max_images_per_dataset,
            "mode": mode,
            "mask_folder": mask_folder,
            "use_butterworth": use_butterworth,
            "use_median": use_median,
            "use_minmax": use_minmax,
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
