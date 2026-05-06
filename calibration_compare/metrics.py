"""Build comparison dataframes from per-model CSVs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .loader import ModelOutputs

GLOBAL_METRIC_COLS = (
    "n_pixels",
    "ece_reg",
    "calibration_mse",
    "mean_nll",
    "sharpness_mean_sigma",
    "rmse",
    "mae",
    "pearson_logvar_logmae",
    "spearman_logvar_logmae",
    "auroc_pixel_top10pct",
    "aurc_pixel",
    "coverage_at_0.50",
    "coverage_at_0.68",
    "coverage_at_0.90",
    "coverage_at_0.95",
    "coverage_at_0.99",
)

PER_MARKER_METRIC_COLS = (
    "n_pixels",
    "ece_reg",
    "sharpness_mean_sigma",
    "mean_nll",
    "mae",
    "rmse",
    "pearson_logvar_logmae",
    "spearman_logvar_logmae",
    "auroc_pixel_top10pct",
    "aurc_pixel",
)


def build_compare_summary(models: list[ModelOutputs], reference_idx: int = 0) -> pd.DataFrame:
    """One row per global metric, columns per model + delta vs reference."""
    rows = []
    for col in GLOBAL_METRIC_COLS:
        row = {"metric": col}
        ref_val = None
        for i, m in enumerate(models):
            g = m.global_metrics[m.global_metrics["group_type"] == "global"].iloc[0]
            v = float(g[col]) if col in g else float("nan")
            row[f"{m.label}"] = v
            if i == reference_idx:
                ref_val = v
        for i, m in enumerate(models):
            if i == reference_idx:
                continue
            row[f"delta__{m.label}__minus__{models[reference_idx].label}"] = (
                row[m.label] - ref_val if ref_val is not None else float("nan")
            )
        rows.append(row)

    # F1 at each quantile (global)
    for m in models:
        bm = m.binary_mask[
            (m.binary_mask["group_type"] == "global") & (m.binary_mask["group_value"] == "all")
        ]
        for _, r in bm.iterrows():
            q = float(r["quantile"])
            metric = f"f1_global_q{int(round(q*100)):02d}"
            existing = next((row for row in rows if row["metric"] == metric), None)
            if existing is None:
                existing = {"metric": metric}
                rows.append(existing)
            existing[m.label] = float(r["f1"])
    # fill deltas for f1 metrics
    ref_label = models[reference_idx].label
    for row in rows:
        if not row["metric"].startswith("f1_global_"):
            continue
        ref_v = row.get(ref_label, float("nan"))
        for i, m in enumerate(models):
            if i == reference_idx:
                continue
            row[f"delta__{m.label}__minus__{ref_label}"] = row.get(m.label, float("nan")) - ref_v

    # AUSE rows (rmse + mae, raw + normalised) drawn from each model's ause.csv.
    for m in models:
        if m.ause is None or m.ause.empty:
            continue
        g = m.ause[m.ause["group_type"] == "global"]
        for _, r in g.iterrows():
            metric = str(r["metric"])
            for stem, val_col in (
                (f"ause_{metric}", "ause"),
                (f"ause_{metric}_normalised", "ause_normalised"),
            ):
                existing = next((row for row in rows if row["metric"] == stem), None)
                if existing is None:
                    existing = {"metric": stem}
                    rows.append(existing)
                existing[m.label] = float(r[val_col])
    for row in rows:
        if not row["metric"].startswith("ause_"):
            continue
        ref_v = row.get(ref_label, float("nan"))
        for i, m in enumerate(models):
            if i == reference_idx:
                continue
            row[f"delta__{m.label}__minus__{ref_label}"] = row.get(m.label, float("nan")) - ref_v

    return pd.DataFrame(rows)


def build_ause_long(models: list[ModelOutputs]) -> pd.DataFrame:
    """Long-format sparsification curves across models."""
    parts = []
    for m in models:
        if m.sparsification_curves is None or m.sparsification_curves.empty:
            continue
        df = m.sparsification_curves.copy()
        df["label"] = m.label
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_ause_summary(models: list[ModelOutputs]) -> pd.DataFrame:
    """Long-format AUSE summary across models (one row per (label, group, metric))."""
    parts = []
    for m in models:
        if m.ause is None or m.ause.empty:
            continue
        df = m.ause.copy()
        df["label"] = m.label
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_compare_per_marker(
    models: list[ModelOutputs],
    markers: list[str],
    reference_idx: int = 0,
) -> pd.DataFrame:
    """Wide table: one row per marker, columns of <metric>__<label> and deltas."""
    out = pd.DataFrame({"marker": markers})
    ref_label = models[reference_idx].label

    # Per-marker rows from global_metrics
    for m in models:
        pm = m.global_metrics[m.global_metrics["group_type"] == "per_marker"].copy()
        pm = pm[pm["group_value"].isin(markers)]
        for col in PER_MARKER_METRIC_COLS:
            out = out.merge(
                pm[["group_value", col]].rename(
                    columns={"group_value": "marker", col: f"{col}__{m.label}"}
                ),
                on="marker",
                how="left",
            )

    # F1 per marker per quantile from binary_mask
    for m in models:
        bm = m.binary_mask[m.binary_mask["group_type"] == "per_marker"].copy()
        bm = bm[bm["group_value"].isin(markers)]
        for q, qdf in bm.groupby("quantile"):
            qtag = f"q{int(round(float(q)*100)):02d}"
            out = out.merge(
                qdf[["group_value", "f1"]].rename(
                    columns={"group_value": "marker", "f1": f"f1_{qtag}__{m.label}"}
                ),
                on="marker",
                how="left",
            )

    # Compute deltas for every metric column relative to the reference label.
    metric_stems: list[str] = []
    for col in PER_MARKER_METRIC_COLS:
        metric_stems.append(col)
    for q in (0.90, 0.95, 0.99):
        metric_stems.append(f"f1_q{int(round(q*100)):02d}")

    for stem in metric_stems:
        ref_col = f"{stem}__{ref_label}"
        if ref_col not in out.columns:
            continue
        for i, m in enumerate(models):
            if i == reference_idx:
                continue
            other_col = f"{stem}__{m.label}"
            if other_col not in out.columns:
                continue
            out[f"delta_{stem}__{m.label}__minus__{ref_label}"] = (
                out[other_col] - out[ref_col]
            )
    return out


def build_reliability_long(models: list[ModelOutputs], group_type: str = "global") -> pd.DataFrame:
    parts = []
    for m in models:
        df = m.coverage_curves[m.coverage_curves["group_type"] == group_type].copy()
        df["label"] = m.label
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def build_f1_vs_q_long(models: list[ModelOutputs]) -> pd.DataFrame:
    parts = []
    for m in models:
        df = m.binary_mask[
            (m.binary_mask["group_type"] == "global") & (m.binary_mask["group_value"] == "all")
        ][["quantile", "f1", "precision", "recall", "iou", "pool_n_pixels"]].copy()
        df["label"] = m.label
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def build_paper_correlation_long(models: list[ModelOutputs]) -> pd.DataFrame:
    parts = []
    for m in models:
        df = m.paper_correlation[
            np.isfinite(m.paper_correlation["log_var_summary"])
            & np.isfinite(m.paper_correlation["log_mae_summary"])
        ].copy()
        df["label"] = m.label
        parts.append(df)
    return pd.concat(parts, ignore_index=True)
