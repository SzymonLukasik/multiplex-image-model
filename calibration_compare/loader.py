"""Load per-model calibration CSVs produced by `python -m calibration.cli`."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from calibration.auroc_subsample import aurc as _aurc_from_arrays
from calibration.config import (
    ALPHA_GRID,
    BINARY_MASK_QUANTILES,
    MIN_PIXELS_PER_MARKER_AUROC,
    REPORT_ALPHAS,
    alpha_col,
)


@dataclass
class ModelOutputs:
    label: str
    model_id: str
    output_dir: Path
    global_metrics: pd.DataFrame
    coverage_curves: pd.DataFrame
    paper_correlation: pd.DataFrame
    binary_mask: pd.DataFrame
    run_metadata: dict

    @property
    def csv_dir(self) -> Path:
        return self.output_dir / "csv"


def load_model(
    output_dir: Path,
    label: str | None = None,
    restrict_dataset: str | None = None,
) -> ModelOutputs:
    """Load all CSVs for one model.

    If ``restrict_dataset`` is set, the model's "global" rows in
    ``global_metrics`` / ``coverage_curves`` / ``binary_mask`` are replaced
    by the matching ``per_dataset=<name>`` rows, and the per-marker rows
    are re-aggregated **from scratch** out of ``per_patch_channel.csv``
    filtered to ``dataset_name == restrict_dataset``. ``paper_correlation``
    is filtered the same way. This makes a multi-dataset model directly
    comparable to a single-dataset one on a fair slice of the data.
    """
    csv_dir = output_dir / "csv"
    if not csv_dir.exists():
        raise FileNotFoundError(f"missing csv/ subdir under {output_dir}")

    required = {
        "global_metrics": csv_dir / "global_metrics.csv",
        "coverage_curves": csv_dir / "coverage_curves.csv",
        "paper_correlation": csv_dir / "paper_correlation.csv",
        "binary_mask": csv_dir / "binary_mask_analysis.csv",
        "run_metadata": csv_dir / "run_metadata.json",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing files for {output_dir}: {missing}")

    with open(required["run_metadata"]) as f:
        meta = json.load(f)
    model_id = str(meta.get("model_id", output_dir.name))

    global_metrics = pd.read_csv(required["global_metrics"])
    coverage_curves = pd.read_csv(required["coverage_curves"])
    paper_correlation = pd.read_csv(required["paper_correlation"])
    binary_mask = pd.read_csv(required["binary_mask"])

    if restrict_dataset:
        global_metrics, coverage_curves, paper_correlation, binary_mask = _restrict_to_dataset(
            output_dir=output_dir,
            csv_dir=csv_dir,
            model_id=model_id,
            global_metrics=global_metrics,
            coverage_curves=coverage_curves,
            paper_correlation=paper_correlation,
            binary_mask=binary_mask,
            dataset_name=restrict_dataset,
        )
        meta = dict(meta)
        meta["restrict_dataset"] = restrict_dataset

    return ModelOutputs(
        label=label or model_id,
        model_id=model_id,
        output_dir=output_dir,
        global_metrics=global_metrics,
        coverage_curves=coverage_curves,
        paper_correlation=paper_correlation,
        binary_mask=binary_mask,
        run_metadata=meta,
    )


def parse_model_arg(spec: str) -> tuple[Path, str | None]:
    """Accept 'path' or 'label:path' (label may not contain ':' or use 'label=path')."""
    if "=" in spec:
        label, path = spec.split("=", 1)
        return Path(path), label
    if ":" in spec and not spec.startswith("/") and Path(spec.split(":", 1)[1]).exists():
        label, path = spec.split(":", 1)
        return Path(path), label
    return Path(spec), None


def load_pool(model: ModelOutputs) -> pd.DataFrame | None:
    """Load a model's AUROC pixel pool (parquet preferred, gzipped CSV fallback)."""
    parquet = model.csv_dir / "auroc_pool.parquet"
    csvgz = model.csv_dir / "auroc_pool.csv.gz"
    if parquet.exists():
        try:
            return pd.read_parquet(parquet)
        except Exception:
            pass
    if csvgz.exists():
        return pd.read_csv(csvgz)
    return None


# ---------------------------------------------------------------------------
# Per-dataset restriction (re-aggregation from per_patch_channel.csv)
# ---------------------------------------------------------------------------

def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    w_sum = float(weights.sum())
    if w_sum <= 0:
        return float("nan")
    return float((values * weights).sum() / w_sum)


def _ece_reg(cov_by_alpha: dict[float, float]) -> tuple[float, float]:
    diffs = np.array([cov_by_alpha[float(a)] - float(a) for a in ALPHA_GRID])
    return float(np.mean(np.abs(diffs))), float(np.mean(diffs * diffs))


def _coverage_curve_from_counts(df_g: pd.DataFrame) -> dict[float, float]:
    """Group-level empirical coverage at each α from per-pc counts."""
    n_total = float(df_g["n_pixels"].sum())
    if n_total <= 0:
        return {float(a): float("nan") for a in ALPHA_GRID}
    cov = {}
    for a in ALPHA_GRID:
        col = alpha_col(a)
        cov[float(a)] = float(df_g[col].sum() / n_total)
    return cov


def _aggregate_global_row(per_pc: pd.DataFrame, model_id: str, group_value: str) -> dict:
    """Re-aggregate one row of `global_metrics` from a filtered per-pc slice."""
    df_g = per_pc
    n_pix = int(df_g["n_pixels"].sum())
    n_pc = int(len(df_g))
    cov = _coverage_curve_from_counts(df_g)
    ece_reg, calib_mse = _ece_reg(cov)

    weights = df_g["n_pixels"].to_numpy(np.float64)
    mean_nll = _weighted_mean(df_g["mean_nll"].to_numpy(np.float64), weights)
    sharpness = _weighted_mean(df_g["mean_sigma"].to_numpy(np.float64), weights)
    mean_var = _weighted_mean(df_g["mean_var"].to_numpy(np.float64), weights)
    rmse = float(np.sqrt(_weighted_mean(df_g["mse"].to_numpy(np.float64), weights)))
    mae = _weighted_mean(df_g["mae"].to_numpy(np.float64), weights)

    valid = (
        np.isfinite(df_g["log_var_summary"]) & np.isfinite(df_g["log_mae_summary"])
    )
    if int(valid.sum()) >= 3:
        pr = pearsonr(df_g.loc[valid, "log_var_summary"], df_g.loc[valid, "log_mae_summary"])
        sr = spearmanr(df_g.loc[valid, "log_var_summary"], df_g.loc[valid, "log_mae_summary"])
        pearson_r = float(pr.statistic)
        pearson_p = float(pr.pvalue)
        spearman_r = float(sr.statistic)
        spearman_p = float(sr.pvalue)
    else:
        pearson_r = pearson_p = spearman_r = spearman_p = float("nan")

    row = {
        "model_id": model_id,
        "group_type": "global",
        "group_value": str(group_value),
        "n_pixels": n_pix,
        "n_patch_channels": n_pc,
        "ece_reg": ece_reg,
        "calibration_mse": calib_mse,
        "mean_nll": mean_nll,
        "sharpness_mean_sigma": sharpness,
        "mean_var": mean_var,
        "rmse": rmse,
        "mae": mae,
        "pearson_logvar_logmae": pearson_r,
        "pearson_pvalue": pearson_p,
        "spearman_logvar_logmae": spearman_r,
        "spearman_pvalue": spearman_p,
        # AUROC/AURC require the pixel pool; fill with NaN here. Comparison
        # plots that need it can re-compute from the (filtered) pool.
        "auroc_pixel_top10pct": float("nan"),
        "aurc_pixel": float("nan"),
        "pool_n_pixels": 0,
    }
    for a in REPORT_ALPHAS:
        row[f"coverage_at_{a:.2f}"] = cov.get(float(round(a, 3)), float("nan"))
    return row


def _build_aggregated_global_metrics(
    per_pc: pd.DataFrame, model_id: str
) -> pd.DataFrame:
    """Recompute global + per-marker rows from a filtered per-pc table."""
    rows = [_aggregate_global_row(per_pc, model_id, "all")]
    for marker, sub in per_pc.groupby("marker_name", sort=True):
        if len(sub) == 0:
            continue
        r = _aggregate_global_row(sub, model_id, str(marker))
        r["group_type"] = "per_marker"
        rows.append(r)
    return pd.DataFrame.from_records(rows)


def _build_aggregated_coverage(per_pc: pd.DataFrame, model_id: str) -> pd.DataFrame:
    """Long-format coverage curves (global + per-marker) for the filtered slice."""
    rows = []
    n_total = int(per_pc["n_pixels"].sum())
    cov_g = _coverage_curve_from_counts(per_pc)
    for a in ALPHA_GRID:
        rows.append(
            {
                "model_id": model_id,
                "group_type": "global",
                "group_value": "all",
                "alpha": float(a),
                "empirical_coverage": cov_g[float(a)],
                "n_pixels": n_total,
                "n_patch_channels": int(len(per_pc)),
            }
        )
    for marker, sub in per_pc.groupby("marker_name", sort=True):
        if len(sub) == 0:
            continue
        cov_m = _coverage_curve_from_counts(sub)
        n_m = int(sub["n_pixels"].sum())
        for a in ALPHA_GRID:
            rows.append(
                {
                    "model_id": model_id,
                    "group_type": "per_marker",
                    "group_value": str(marker),
                    "alpha": float(a),
                    "empirical_coverage": cov_m[float(a)],
                    "n_pixels": n_m,
                    "n_patch_channels": int(len(sub)),
                }
            )
    return pd.DataFrame.from_records(rows)


def _build_aggregated_binary_mask(
    output_dir: Path, model_id: str, dataset_name: str
) -> pd.DataFrame | None:
    """Re-aggregate F1/IoU at each quantile from the filtered pixel pool."""
    pool_paths = [
        output_dir / "csv" / "auroc_pool.parquet",
        output_dir / "csv" / "auroc_pool.csv.gz",
    ]
    pool: pd.DataFrame | None = None
    for p in pool_paths:
        if p.exists():
            try:
                pool = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
                break
            except Exception:
                continue
    if pool is None or pool.empty:
        print(f"[restrict] {output_dir}: pool missing/empty -> binary_mask not re-aggregated")
        return None
    if "dataset" not in pool.columns:
        # Older pool schema. Try to recover when the originating per_patch
        # CSV is single-dataset and matches the requested slice.
        per_pc_path = output_dir / "csv" / "per_patch_channel.csv"
        if per_pc_path.exists():
            ds_seen = pd.read_csv(per_pc_path, usecols=["dataset_name"])["dataset_name"].unique()
            if len(ds_seen) == 1 and str(ds_seen[0]) == dataset_name:
                print(
                    f"[restrict] {output_dir}: pool lacks 'dataset' column but the run is "
                    f"single-dataset ({dataset_name!r}); treating the whole pool as that slice."
                )
                pool = pool.assign(dataset=dataset_name)
            else:
                print(
                    f"[restrict] {output_dir}: pool lacks 'dataset' column and run spans "
                    f"{len(ds_seen)} datasets -> binary_mask not re-aggregated. "
                    f"Re-run run_calibration.py to regenerate the pool with dataset tags."
                )
                return None
        else:
            print(f"[restrict] {output_dir}: pool lacks 'dataset' column -> binary_mask not re-aggregated")
            return None
    pool_g = pool[pool["dataset"] == dataset_name]
    if len(pool_g) < MIN_PIXELS_PER_MARKER_AUROC:
        print(
            f"[restrict] {output_dir}: pool has only {len(pool_g)} rows for "
            f"dataset={dataset_name!r} (min={MIN_PIXELS_PER_MARKER_AUROC}) -> "
            f"binary_mask not re-aggregated"
        )
        return None

    rows = []

    def _row(group_type: str, group_value: str, sub: pd.DataFrame) -> list[dict]:
        if len(sub) < MIN_PIXELS_PER_MARKER_AUROC:
            return []
        sigma = sub["sigma"].to_numpy(np.float64)
        abs_r = sub["abs_residual"].to_numpy(np.float64)
        out = []
        for q in BINARY_MASK_QUANTILES:
            sigma_thr = float(np.quantile(sigma, q))
            r_thr = float(np.quantile(abs_r, q))
            m_sigma = sigma > sigma_thr
            m_r = abs_r > r_thr
            tp = int(np.sum(m_sigma & m_r))
            fp = int(np.sum(m_sigma & ~m_r))
            fn = int(np.sum(~m_sigma & m_r))
            tn = int(len(sub) - tp - fp - fn)
            precision = tp / (tp + fp) if (tp + fp) else float("nan")
            recall = tp / (tp + fn) if (tp + fn) else float("nan")
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision and recall and (precision + recall) > 0
                else float("nan")
            )
            iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
            out.append(
                {
                    "model_id": model_id,
                    "group_type": group_type,
                    "group_value": group_value,
                    "quantile": q,
                    "sigma_threshold": sigma_thr,
                    "abs_r_threshold": r_thr,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "iou": iou,
                    "pool_n_pixels": int(len(sub)),
                }
            )
        return out

    rows.extend(_row("global", "all", pool_g))
    for marker, sub in pool_g.groupby("marker"):
        rows.extend(_row("per_marker", str(marker), sub))
    return pd.DataFrame.from_records(rows)


def _binary_mask_from_per_pc(
    per_pc: pd.DataFrame, model_id: str
) -> pd.DataFrame:
    """Fallback: re-aggregate F1/IoU at each quantile by summing the per-patch
    tp_q*/fp_q*/fn_q*/tn_q* columns already present in `per_patch_channel.csv`.

    Note: thresholds are *per-patch* (within-patch quantiles), not a single
    HN-global threshold like the pool-based path. Conservative but consistent
    fallback when the AUROC pool is missing the dataset column or the file.
    """
    rows: list[dict] = []

    def _f1_iou(tp: int, fp: int, fn: int, tn: int) -> tuple[float, float, float, float]:
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision and recall and (precision + recall) > 0
            else float("nan")
        )
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
        return precision, recall, f1, iou

    def _emit(group_type: str, group_value: str, df_g: pd.DataFrame) -> list[dict]:
        out = []
        for q in BINARY_MASK_QUANTILES:
            qtag = f"{q:.2f}"
            cols = (f"tp_q{qtag}", f"fp_q{qtag}", f"fn_q{qtag}", f"tn_q{qtag}")
            if any(c not in df_g.columns for c in cols):
                continue
            tp = int(df_g[cols[0]].sum())
            fp = int(df_g[cols[1]].sum())
            fn = int(df_g[cols[2]].sum())
            tn = int(df_g[cols[3]].sum())
            precision, recall, f1, iou = _f1_iou(tp, fp, fn, tn)
            n_pix = int(df_g["n_pixels"].sum()) if "n_pixels" in df_g.columns else (tp + fp + fn + tn)
            out.append(
                {
                    "model_id": model_id,
                    "group_type": group_type,
                    "group_value": group_value,
                    "quantile": q,
                    "sigma_threshold": float("nan"),
                    "abs_r_threshold": float("nan"),
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "precision": precision, "recall": recall,
                    "f1": f1, "iou": iou,
                    "pool_n_pixels": n_pix,
                }
            )
        return out

    rows.extend(_emit("global", "all", per_pc))
    for marker, sub in per_pc.groupby("marker_name", sort=True):
        rows.extend(_emit("per_marker", str(marker), sub))
    return pd.DataFrame.from_records(rows)


def _augment_global_with_pool_auroc(
    global_metrics: pd.DataFrame, output_dir: Path, dataset_name: str
) -> pd.DataFrame:
    """Fill auroc_pixel_top10pct / aurc_pixel / pool_n_pixels using the filtered pool."""
    pool_paths = [
        output_dir / "csv" / "auroc_pool.parquet",
        output_dir / "csv" / "auroc_pool.csv.gz",
    ]
    pool: pd.DataFrame | None = None
    for p in pool_paths:
        if p.exists():
            try:
                pool = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
                break
            except Exception:
                continue
    if pool is None or pool.empty:
        return global_metrics
    if "dataset" not in pool.columns:
        per_pc_path = output_dir / "csv" / "per_patch_channel.csv"
        if per_pc_path.exists():
            ds_seen = pd.read_csv(per_pc_path, usecols=["dataset_name"])["dataset_name"].unique()
            if len(ds_seen) == 1 and str(ds_seen[0]) == dataset_name:
                pool = pool.assign(dataset=dataset_name)
            else:
                return global_metrics
        else:
            return global_metrics
    pool_g = pool[pool["dataset"] == dataset_name]

    def _auroc_aurc(sub: pd.DataFrame) -> tuple[float, float, int]:
        n = len(sub)
        if n < MIN_PIXELS_PER_MARKER_AUROC:
            return float("nan"), float("nan"), n
        sigma = sub["sigma"].to_numpy(np.float64)
        abs_r = sub["abs_residual"].to_numpy(np.float64)
        p90 = float(np.quantile(abs_r, 0.90))
        label = (abs_r > p90).astype(np.int8)
        if label.sum() == 0 or label.sum() == n:
            auroc = float("nan")
        else:
            auroc = float(roc_auc_score(label, sigma))
        return auroc, _aurc_from_arrays(sigma, abs_r * abs_r), n

    out = global_metrics.copy()
    for idx, row in out.iterrows():
        if row["group_type"] == "global":
            sub = pool_g
        elif row["group_type"] == "per_marker":
            sub = pool_g[pool_g["marker"] == row["group_value"]]
        else:
            continue
        a, r, n = _auroc_aurc(sub)
        out.at[idx, "auroc_pixel_top10pct"] = a
        out.at[idx, "aurc_pixel"] = r
        out.at[idx, "pool_n_pixels"] = n
    return out


def _restrict_to_dataset(
    output_dir: Path,
    csv_dir: Path,
    model_id: str,
    global_metrics: pd.DataFrame,
    coverage_curves: pd.DataFrame,
    paper_correlation: pd.DataFrame,
    binary_mask: pd.DataFrame,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replace each frame's "global"/"per_marker" rows with HN-restricted ones."""
    per_pc_path = csv_dir / "per_patch_channel.csv"
    if not per_pc_path.exists():
        raise FileNotFoundError(
            f"--restrict-dataset requires {per_pc_path} for re-aggregation"
        )
    # Per-patch-channel rows have ~150 columns; only load what we need.
    needed_base = {
        "image_path",
        "dataset_name",
        "marker_name",
        "channel_id",
        "n_pixels",
        "mse",
        "mae",
        "rmse",
        "mean_var",
        "mean_sigma",
        "mean_logvar",
        "mean_nll",
        "log_mae_summary",
        "log_var_summary",
    }
    bm_prefixes = ("tp_q", "fp_q", "fn_q", "tn_q")
    per_pc = pd.read_csv(
        per_pc_path,
        usecols=lambda c: (
            c in needed_base
            or c.startswith("cov_count_")
            or c.startswith(bm_prefixes)
        ),
    )
    per_pc = per_pc[per_pc["dataset_name"] == dataset_name]
    if per_pc.empty:
        raise ValueError(
            f"no per_patch_channel rows for dataset={dataset_name} in {per_pc_path}"
        )

    # 1) global_metrics: re-aggregate global + per-marker; fill AUROC from pool.
    new_global = _build_aggregated_global_metrics(per_pc, model_id=model_id)
    new_global = _augment_global_with_pool_auroc(
        new_global, output_dir=output_dir, dataset_name=dataset_name
    )
    # Keep any original per_dataset rows so downstream code that may inspect
    # them still works; just drop the old "global" + "per_marker" rows.
    keep_other = global_metrics[
        ~global_metrics["group_type"].isin(["global", "per_marker"])
    ]
    global_metrics_out = pd.concat([new_global, keep_other], ignore_index=True)

    # 2) coverage_curves: same trick — global + per-marker recomputed.
    new_cov = _build_aggregated_coverage(per_pc, model_id=model_id)
    keep_cov = coverage_curves[
        ~coverage_curves["group_type"].isin(["global", "per_marker"])
    ]
    coverage_out = pd.concat([new_cov, keep_cov], ignore_index=True)

    # 3) binary_mask: prefer pool-based (HN-global thresholds); fall back to
    #    summing per-pc TP/FP/FN/TN columns (within-patch thresholds) when the
    #    pool is missing/old. The fallback uses different thresholds but is
    #    still informative and lets every model render.
    new_bm = _build_aggregated_binary_mask(
        output_dir=output_dir, model_id=model_id, dataset_name=dataset_name
    )
    used_fallback = False
    if new_bm is None or new_bm.empty:
        new_bm = _binary_mask_from_per_pc(per_pc, model_id=model_id)
        used_fallback = bool(len(new_bm))
        if used_fallback:
            print(
                f"[restrict] {output_dir}: binary_mask fallback to per-patch TP/FP "
                f"sums (within-patch thresholds — pool was unavailable)."
            )
    keep_bm = binary_mask[
        ~binary_mask["group_type"].isin(["global", "per_marker"])
    ]
    if new_bm is not None and len(new_bm):
        binary_mask_out = pd.concat([new_bm, keep_bm], ignore_index=True)
    else:
        binary_mask_out = keep_bm.reset_index(drop=True)

    # 4) paper_correlation: filter to dataset.
    if "dataset_name" in paper_correlation.columns:
        paper_out = paper_correlation[
            paper_correlation["dataset_name"] == dataset_name
        ].reset_index(drop=True)
    else:
        paper_out = paper_correlation

    return global_metrics_out, coverage_out, paper_out, binary_mask_out


def assert_marker_overlap(models: list[ModelOutputs]) -> list[str]:
    """Return the intersection of per-marker group_values across all models; warn on diffs."""
    marker_sets = []
    for m in models:
        pm = m.global_metrics[m.global_metrics["group_type"] == "per_marker"]
        marker_sets.append(set(pm["group_value"].astype(str).tolist()))
    intersection = set.intersection(*marker_sets) if marker_sets else set()
    union = set.union(*marker_sets) if marker_sets else set()
    diff = union - intersection
    if diff:
        for m, s in zip(models, marker_sets):
            only = sorted(s - intersection)
            if only:
                print(f"[compare] WARN: markers only in '{m.label}': {only}")
    return sorted(intersection)
