"""Group-level aggregation of per-(patch, channel) metrics.

Reads `per_patch_channel.csv` and produces:
  - coverage_curves.csv      (long format, per (group_type, group_value, alpha))
  - global_metrics.csv       (one row per (group_type, group_value))
  - paper_correlation.csv    (Figure 5 inputs and r summary)
  - binary_mask_analysis.csv (group-level F1/IoU at 3 quantile thresholds)
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from .auroc_subsample import aurc
from .config import (
    ALPHA_GRID,
    BINARY_MASK_QUANTILES,
    MIN_PIXELS_PER_MARKER_AUROC,
    REPORT_ALPHAS,
    alpha_col,
    quantile_cols,
)
from .metrics_pixel import ause as _ause


AUSE_METRICS = ("rmse", "mae")
AUSE_N_POINTS = 100


GROUPINGS = (
    ("global", lambda df: [("all", df)]),
    ("per_marker", lambda df: list(df.groupby("marker_name", sort=True))),
    ("per_dataset", lambda df: list(df.groupby("dataset_name", sort=True))),
)


def _coverage_curve(df_group: pd.DataFrame) -> dict[float, float]:
    """Group-level empirical coverage at each α from per-pc counts."""
    n_total = float(df_group["n_pixels"].sum())
    if n_total <= 0:
        return {a: float("nan") for a in ALPHA_GRID}
    cov = {}
    for a in ALPHA_GRID:
        col = alpha_col(a)
        cov[float(a)] = float(df_group[col].sum() / n_total)
    return cov


def _ece_reg(cov: dict[float, float]) -> tuple[float, float]:
    """Mean and squared deviation of empirical coverage from nominal α."""
    diffs = np.array([cov[float(a)] - float(a) for a in ALPHA_GRID])
    return float(np.mean(np.abs(diffs))), float(np.mean(diffs * diffs))


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    w_sum = float(weights.sum())
    if w_sum <= 0:
        return float("nan")
    return float((values * weights).sum() / w_sum)


def build_coverage_curves(per_pc: pd.DataFrame, model_id: str) -> pd.DataFrame:
    rows = []
    for group_type, splitter in GROUPINGS:
        for group_value, df_g in splitter(per_pc):
            cov = _coverage_curve(df_g)
            for a in ALPHA_GRID:
                rows.append(
                    {
                        "model_id": model_id,
                        "group_type": group_type,
                        "group_value": str(group_value),
                        "alpha": float(a),
                        "empirical_coverage": cov[float(a)],
                        "n_pixels": int(df_g["n_pixels"].sum()),
                        "n_patch_channels": int(len(df_g)),
                    }
                )
    return pd.DataFrame.from_records(rows)


def build_global_metrics(
    per_pc: pd.DataFrame,
    pool: pd.DataFrame,
    model_id: str,
) -> pd.DataFrame:
    rows = []
    for group_type, splitter in GROUPINGS:
        for group_value, df_g in splitter(per_pc):
            n_pix = float(df_g["n_pixels"].sum())
            n_pc = int(len(df_g))
            cov = _coverage_curve(df_g)
            ece_reg, calib_mse = _ece_reg(cov)

            weights = df_g["n_pixels"].to_numpy(np.float64)
            mean_nll = _weighted_mean(df_g["mean_nll"].to_numpy(np.float64), weights)
            sharpness = _weighted_mean(df_g["mean_sigma"].to_numpy(np.float64), weights)
            mean_var = _weighted_mean(df_g["mean_var"].to_numpy(np.float64), weights)
            rmse = float(
                np.sqrt(_weighted_mean(df_g["mse"].to_numpy(np.float64), weights))
            )
            mae = _weighted_mean(df_g["mae"].to_numpy(np.float64), weights)

            # Pearson / Spearman: per-(patch, channel) log_var vs log_MAE.
            valid = (
                np.isfinite(df_g["log_var_summary"])
                & np.isfinite(df_g["log_mae_summary"])
            )
            if int(valid.sum()) >= 3:
                pr = pearsonr(
                    df_g.loc[valid, "log_var_summary"],
                    df_g.loc[valid, "log_mae_summary"],
                )
                sr = spearmanr(
                    df_g.loc[valid, "log_var_summary"],
                    df_g.loc[valid, "log_mae_summary"],
                )
                pearson_r = float(pr.statistic)
                pearson_p = float(pr.pvalue)
                spearman_r = float(sr.statistic)
                spearman_p = float(sr.pvalue)
            else:
                pearson_r = pearson_p = spearman_r = spearman_p = float("nan")

            # Pixel-level AUROC / AURC from the subsample pool.
            pool_g = _slice_pool(pool, group_type, group_value)
            auroc_val, aurc_val, pool_n = _pool_auroc_aurc(pool_g)

            row = {
                "model_id": model_id,
                "group_type": group_type,
                "group_value": str(group_value),
                "n_pixels": int(n_pix),
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
                "auroc_pixel_top10pct": auroc_val,
                "aurc_pixel": aurc_val,
                "pool_n_pixels": pool_n,
            }
            for a in REPORT_ALPHAS:
                row[f"coverage_at_{a:.2f}"] = cov.get(float(round(a, 3)), float("nan"))
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def _slice_pool(
    pool: pd.DataFrame, group_type: str, group_value: str
) -> pd.DataFrame:
    if group_type == "global":
        return pool
    if group_type == "per_marker":
        return pool[pool["marker"] == group_value]
    if group_type == "per_dataset":
        return pool[pool["dataset"] == group_value]
    raise ValueError(group_type)


def _pool_auroc_aurc(pool_g: pd.DataFrame) -> tuple[float, float, int]:
    n = len(pool_g)
    if n < MIN_PIXELS_PER_MARKER_AUROC:
        return float("nan"), float("nan"), n
    sigma = pool_g["sigma"].to_numpy(np.float64)
    abs_r = pool_g["abs_residual"].to_numpy(np.float64)
    p90 = float(np.quantile(abs_r, 0.90))
    label = (abs_r > p90).astype(np.int8)
    if label.sum() == 0 or label.sum() == n:
        auroc_val = float("nan")
    else:
        auroc_val = float(roc_auc_score(label, sigma))
    aurc_val = aurc(sigma, abs_r * abs_r)
    return auroc_val, aurc_val, n


def build_paper_correlation(per_pc: pd.DataFrame, model_id: str) -> pd.DataFrame:
    """Per-(patch, channel) Figure-5 inputs."""
    cols = [
        "image_index",
        "image_path",
        "dataset_name",
        "marker_name",
        "channel_id",
        "n_pixels",
        "log_mae_summary",
        "log_var_summary",
        "mae",
        "mean_var",
    ]
    out = per_pc.loc[:, cols].copy()
    out.insert(0, "model_id", model_id)
    return out


def build_binary_mask_analysis(
    per_pc: pd.DataFrame, pool: pd.DataFrame, model_id: str
) -> pd.DataFrame:
    """Group-level F1/IoU using thresholds computed from the global pool slice."""
    rows = []
    for group_type, splitter in GROUPINGS:
        for group_value, df_g in splitter(per_pc):
            pool_g = _slice_pool(pool, group_type, group_value)
            n_pool = len(pool_g)
            if n_pool < MIN_PIXELS_PER_MARKER_AUROC:
                continue
            sigma = pool_g["sigma"].to_numpy(np.float64)
            abs_r = pool_g["abs_residual"].to_numpy(np.float64)
            for q in BINARY_MASK_QUANTILES:
                sigma_thr = float(np.quantile(sigma, q))
                r_thr = float(np.quantile(abs_r, q))
                m_sigma = sigma > sigma_thr
                m_r = abs_r > r_thr
                tp = int(np.sum(m_sigma & m_r))
                fp = int(np.sum(m_sigma & ~m_r))
                fn = int(np.sum(~m_sigma & m_r))
                tn = int(n_pool - tp - fp - fn)
                precision = tp / (tp + fp) if (tp + fp) else float("nan")
                recall = tp / (tp + fn) if (tp + fn) else float("nan")
                f1 = (
                    2.0 * precision * recall / (precision + recall)
                    if (precision and recall and (precision + recall) > 0)
                    else float("nan")
                )
                iou = (
                    tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
                )
                rows.append(
                    {
                        "model_id": model_id,
                        "group_type": group_type,
                        "group_value": str(group_value),
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
                        "pool_n_pixels": n_pool,
                    }
                )
    return pd.DataFrame.from_records(rows)


# ---------------------------------------------------------------------------
# AUSE — sparsification-error area, computed from the AUROC pool
# ---------------------------------------------------------------------------

def _ause_groups(pool: pd.DataFrame):
    """Yield (group_type, group_value, sub_df) for AUSE aggregation."""
    yield "global", "all", pool
    if "marker" in pool.columns:
        for marker, sub in pool.groupby("marker"):
            yield "per_marker", str(marker), sub


def build_ause(pool: pd.DataFrame, model_id: str, seed: int = 0) -> tuple[
    pd.DataFrame, pd.DataFrame
]:
    """Compute AUSE (RMSE & MAE) at global + per-marker level.

    Returns `(ause_df, sparsification_curves_df)`. Per-marker groups with
    fewer than `MIN_PIXELS_PER_MARKER_AUROC` pixels are skipped (consistent
    with AUROC).
    """
    rows_summary: list[dict] = []
    rows_curves: list[dict] = []
    for group_type, group_value, sub in _ause_groups(pool):
        n = len(sub)
        if group_type == "per_marker" and n < MIN_PIXELS_PER_MARKER_AUROC:
            continue
        if n < 2:
            continue
        sigma = sub["sigma"].to_numpy(dtype=np.float64)
        abs_r = sub["abs_residual"].to_numpy(dtype=np.float64)
        for metric in AUSE_METRICS:
            res = _ause(sigma, abs_r, metric=metric, n_points=AUSE_N_POINTS, seed=seed)
            rows_summary.append(
                {
                    "model_id": model_id,
                    "group_type": group_type,
                    "group_value": group_value,
                    "metric": metric,
                    "ause": res["ause"],
                    "ause_random": res["ause_random"],
                    "ause_normalised": res["ause_normalised"],
                    "n_pixels": int(n),
                }
            )
            for f, e_s, e_o, e_r, se in zip(
                res["fractions"],
                res["err_sigma"],
                res["err_oracle"],
                res["err_random"],
                res["sparsification_error"],
            ):
                rows_curves.append(
                    {
                        "model_id": model_id,
                        "group_type": group_type,
                        "group_value": group_value,
                        "metric": metric,
                        "fraction": float(f),
                        "err_sigma": float(e_s),
                        "err_oracle": float(e_o),
                        "err_random": float(e_r),
                        "sparsification_error": float(se),
                    }
                )
    ause_df = pd.DataFrame.from_records(rows_summary)
    curves_df = pd.DataFrame.from_records(rows_curves)
    return ause_df, curves_df
