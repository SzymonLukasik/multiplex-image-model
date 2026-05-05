"""Comparison figures (N models). Read CSVs and dataframes; write PNG + PDF."""
from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

from .loader import ModelOutputs, load_pool

DEFAULT_PALETTE = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#17becf",
    "#e377c2",
)


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _colors(n: int) -> list[str]:
    if n <= len(DEFAULT_PALETTE):
        return list(DEFAULT_PALETTE[:n])
    cmap = plt.get_cmap("tab20")
    return [cmap(i / max(1, n - 1)) for i in range(n)]


def reliability_global_compare(
    models: list[ModelOutputs], out_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="ideal")
    colors = _colors(len(models))
    for m, color in zip(models, colors):
        cov = m.coverage_curves[m.coverage_curves["group_type"] == "global"].sort_values("alpha")
        g = m.global_metrics[m.global_metrics["group_type"] == "global"]
        ece = float(g["ece_reg"].iloc[0]) if len(g) else float("nan")
        ax.plot(
            cov["alpha"],
            cov["empirical_coverage"],
            color=color,
            linewidth=2,
            label=f"{m.label}  (ECE={ece:.3f})",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("nominal coverage α")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Reliability — global (LOO)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "reliability_global_compare")


def reliability_per_marker_compare(
    models: list[ModelOutputs], markers: list[str], out_dir: Path
) -> None:
    if not markers:
        return
    # Sort by max ECE across models to surface worst first.
    ece_max = []
    for marker in markers:
        vals = []
        for m in models:
            pm = m.global_metrics[
                (m.global_metrics["group_type"] == "per_marker")
                & (m.global_metrics["group_value"] == marker)
            ]
            if len(pm):
                vals.append(float(pm["ece_reg"].iloc[0]))
        ece_max.append(max(vals) if vals else 0.0)
    order = [m for _, m in sorted(zip(ece_max, markers), reverse=True)]

    n = len(order)
    cols = min(6, max(3, int(np.ceil(np.sqrt(n)))))
    rows = ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.4 * rows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes).reshape(rows, cols)

    colors = _colors(len(models))
    for i, marker in enumerate(order):
        ax = axes[i // cols, i % cols]
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=0.8)
        title_bits = [marker]
        for m, color in zip(models, colors):
            cov = m.coverage_curves[
                (m.coverage_curves["group_type"] == "per_marker")
                & (m.coverage_curves["group_value"] == marker)
            ].sort_values("alpha")
            g = m.global_metrics[
                (m.global_metrics["group_type"] == "per_marker")
                & (m.global_metrics["group_value"] == marker)
            ]
            ece = float(g["ece_reg"].iloc[0]) if len(g) else float("nan")
            ax.plot(cov["alpha"], cov["empirical_coverage"], color=color, linewidth=1.2)
            title_bits.append(f"{m.label[:8]}={ece:.2f}")
        ax.set_title(" | ".join(title_bits), fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="both", labelsize=7)

    for i in range(len(order), rows * cols):
        axes[i // cols, i % cols].axis("off")

    handles = [plt.Line2D([0], [0], color=c, linewidth=2) for c in colors]
    fig.legend(handles, [m.label for m in models], loc="upper right", fontsize=8)
    fig.suptitle("Per-marker reliability (sorted by max ECE↓)", fontsize=11)
    fig.text(0.5, 0.04, "nominal coverage α", ha="center", fontsize=10)
    fig.text(0.04, 0.5, "empirical coverage", va="center", rotation="vertical", fontsize=10)
    fig.tight_layout(rect=[0.05, 0.05, 1, 0.97])
    _save(fig, out_dir, "reliability_per_marker_compare")


def ece_per_marker_compare(
    compare_per_marker: pd.DataFrame,
    models: list[ModelOutputs],
    out_dir: Path,
) -> None:
    cols = [f"ece_reg__{m.label}" for m in models]
    df = compare_per_marker.dropna(subset=cols).copy()
    if df.empty:
        return
    df["max_ece"] = df[cols].max(axis=1)
    df = df.sort_values("max_ece", ascending=True)

    colors = _colors(len(models))
    n_markers = len(df)
    bar_h = 0.8 / len(models)
    y = np.arange(n_markers)

    fig, ax = plt.subplots(figsize=(7.0, max(4.0, 0.25 * n_markers)))
    for i, (m, color) in enumerate(zip(models, colors)):
        offset = (i - (len(models) - 1) / 2) * bar_h
        ax.barh(y + offset, df[f"ece_reg__{m.label}"], height=bar_h, color=color, label=m.label)
    ax.set_yticks(y)
    ax.set_yticklabels(df["marker"].tolist(), fontsize=7)
    ax.set_xlabel("ECE_reg")
    ax.set_title("Per-marker ECE — paired bars")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, out_dir, "ece_per_marker_compare")


def ece_scatter_pairwise(
    compare_per_marker: pd.DataFrame,
    models: list[ModelOutputs],
    out_dir: Path,
    reference_idx: int = 0,
) -> None:
    """Scatter of per-marker ECE for each non-reference model vs reference."""
    ref = models[reference_idx]
    others = [m for i, m in enumerate(models) if i != reference_idx]
    if not others:
        return
    n = len(others)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5.0), squeeze=False)
    for ax, m in zip(axes[0], others):
        x = compare_per_marker[f"ece_reg__{ref.label}"]
        y = compare_per_marker[f"ece_reg__{m.label}"]
        valid = x.notna() & y.notna()
        ax.scatter(x[valid], y[valid], color="C0", alpha=0.7)
        lo = float(min(x[valid].min(), y[valid].min())) if valid.any() else 0.0
        hi = float(max(x[valid].max(), y[valid].max())) if valid.any() else 1.0
        pad = (hi - lo) * 0.05 if hi > lo else 0.01
        lo -= pad
        hi += pad
        ax.plot([lo, hi], [lo, hi], color="grey", linestyle="--", linewidth=1, label="y = x")
        # Label the worst-degrading markers
        delta = (y - x).abs()
        if valid.any():
            top = delta[valid].sort_values(ascending=False).head(6).index
            for idx in top:
                ax.annotate(
                    str(compare_per_marker.loc[idx, "marker"]),
                    xy=(x[idx], y[idx]),
                    fontsize=7,
                    xytext=(2, 2),
                    textcoords="offset points",
                )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"ECE — {ref.label}")
        ax.set_ylabel(f"ECE — {m.label}")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{m.label} vs {ref.label}")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Per-marker ECE — pairwise scatter", fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir, "ece_scatter_pairwise")


def ece_delta_per_marker(
    compare_per_marker: pd.DataFrame,
    models: list[ModelOutputs],
    out_dir: Path,
    reference_idx: int = 0,
) -> None:
    ref = models[reference_idx]
    others = [m for i, m in enumerate(models) if i != reference_idx]
    if not others:
        return

    n = len(others)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, max(4.0, 0.22 * len(compare_per_marker))), squeeze=False)
    for ax, m in zip(axes[0], others):
        delta_col = f"delta_ece_reg__{m.label}__minus__{ref.label}"
        if delta_col not in compare_per_marker.columns:
            continue
        df = compare_per_marker[["marker", delta_col]].dropna().sort_values(delta_col)
        colors = ["#d62728" if v > 0 else "#2ca02c" for v in df[delta_col]]
        ax.barh(df["marker"], df[delta_col], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(f"ΔECE = {m.label} − {ref.label}")
        ax.set_title(f"{m.label} − {ref.label}")
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Per-marker ECE delta (positive = degradation)", fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir, "ece_delta_per_marker")


def f1_per_marker_compare(
    compare_per_marker: pd.DataFrame,
    models: list[ModelOutputs],
    out_dir: Path,
    quantiles: tuple[float, ...] = (0.90, 0.95, 0.99),
) -> None:
    fig, axes = plt.subplots(1, len(quantiles), figsize=(6.0 * len(quantiles), max(4.5, 0.22 * len(compare_per_marker))), squeeze=False)
    colors = _colors(len(models))
    for ax, q in zip(axes[0], quantiles):
        qtag = f"q{int(round(q*100)):02d}"
        cols = [f"f1_{qtag}__{m.label}" for m in models]
        if any(c not in compare_per_marker.columns for c in cols):
            ax.set_visible(False)
            continue
        df = compare_per_marker.dropna(subset=cols).copy()
        if df.empty:
            ax.set_visible(False)
            continue
        df = df.sort_values(cols[0], ascending=True)
        n_markers = len(df)
        bar_h = 0.8 / len(models)
        y = np.arange(n_markers)
        for i, (m, color) in enumerate(zip(models, colors)):
            offset = (i - (len(models) - 1) / 2) * bar_h
            ax.barh(y + offset, df[f"f1_{qtag}__{m.label}"], height=bar_h, color=color, label=m.label)
        ax.axvline(1.0 - q, color="black", linestyle="--", linewidth=0.8, label=f"chance ({1.0 - q:.2f})")
        ax.set_yticks(y)
        ax.set_yticklabels(df["marker"].tolist(), fontsize=7)
        ax.set_xlabel(f"F1 (q={q})")
        ax.set_title(f"q = {q}")
        ax.grid(axis="x", alpha=0.3)
        ax.legend(loc="lower right", fontsize=7)
    fig.suptitle("Per-marker σ-vs-|residual| F1", fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir, "f1_per_marker_compare")


def f1_global_vs_q_compare(
    f1_long: pd.DataFrame, models: list[ModelOutputs], out_dir: Path
) -> None:
    if f1_long.empty:
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    colors = _colors(len(models))
    for m, color in zip(models, colors):
        sub = f1_long[f1_long["label"] == m.label].sort_values("quantile")
        ax.plot(sub["quantile"], sub["f1"], color=color, marker="o", label=m.label)
    qs = np.linspace(f1_long["quantile"].min(), f1_long["quantile"].max(), 50)
    ax.plot(qs, 1.0 - qs, color="black", linestyle="--", linewidth=0.8, label="chance (1−q)")
    ax.set_xlabel("quantile q")
    ax.set_ylabel("F1 (global, σ-vs-|r|)")
    ax.set_title("Global F1 vs strictness")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "f1_global_vs_q_compare")


def sharpness_vs_ece_compare(
    compare_per_marker: pd.DataFrame,
    models: list[ModelOutputs],
    out_dir: Path,
    reference_idx: int = 0,
) -> None:
    colors = _colors(len(models))
    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    for m, color in zip(models, colors):
        s_col = f"sharpness_mean_sigma__{m.label}"
        e_col = f"ece_reg__{m.label}"
        if s_col not in compare_per_marker.columns or e_col not in compare_per_marker.columns:
            continue
        ax.scatter(
            compare_per_marker[s_col],
            compare_per_marker[e_col],
            color=color,
            alpha=0.75,
            label=m.label,
        )

    if len(models) == 2:
        ref = models[reference_idx]
        other = models[1 - reference_idx]
        for _, row in compare_per_marker.iterrows():
            x0 = row.get(f"sharpness_mean_sigma__{ref.label}")
            y0 = row.get(f"ece_reg__{ref.label}")
            x1 = row.get(f"sharpness_mean_sigma__{other.label}")
            y1 = row.get(f"ece_reg__{other.label}")
            if any(pd.isna(v) for v in (x0, y0, x1, y1)):
                continue
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color="grey", alpha=0.4, linewidth=0.7),
            )

    ax.set_xlabel("sharpness (mean σ over pixels)")
    ax.set_ylabel("ECE_reg")
    ax.set_title("Sharpness vs ECE per marker")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "sharpness_vs_ece_compare")


def loo_scatter_compare(
    paper_long: pd.DataFrame, models: list[ModelOutputs], out_dir: Path
) -> None:
    if paper_long.empty:
        return
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5.0), squeeze=False, sharex=True, sharey=True)
    colors = _colors(n)
    x_all = paper_long["log_var_summary"].to_numpy()
    y_all = paper_long["log_mae_summary"].to_numpy()
    if x_all.size:
        x_lo, x_hi = float(np.nanmin(x_all)), float(np.nanmax(x_all))
        y_lo, y_hi = float(np.nanmin(y_all)), float(np.nanmax(y_all))
    else:
        x_lo, x_hi, y_lo, y_hi = 0.0, 1.0, 0.0, 1.0

    for ax, m, color in zip(axes[0], models, colors):
        sub = paper_long[paper_long["label"] == m.label]
        x = sub["log_var_summary"].to_numpy()
        y = sub["log_mae_summary"].to_numpy()
        ax.scatter(x, y, s=2, alpha=0.18, color=color, linewidths=0)
        if x.size >= 2:
            lr = linregress(x, y)
            xs = np.linspace(x_lo, x_hi, 200)
            ax.plot(
                xs,
                lr.intercept + lr.slope * xs,
                color="black",
                linewidth=1.4,
                label=f"slope={lr.slope:.2f}",
            )
            r = float(np.corrcoef(x, y)[0, 1])
        else:
            r = float("nan")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(r"$\log\,\overline{\sigma^{2}}$")
        ax.set_ylabel(r"$\log\,\mathrm{MAE}$")
        ax.set_title(f"{m.label}\nPearson r = {r:.3f}")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Figure-5-style LOO scatter (shared axes)", fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir, "loo_scatter_compare")


def _reliability_top_n_grid(
    compare_per_marker: pd.DataFrame,
    models: list[ModelOutputs],
    out_dir: Path,
    reference_idx: int,
    top_n: int,
    *,
    direction: str,
    stem: str,
    title_prefix: str,
) -> None:
    """Shared helper: small-multiples reliability grid for top-N markers by ΔECE.

    direction = "degraded" → sort delta descending (other > ref → larger ECE);
    direction = "improved" → sort delta ascending (other < ref → smaller ECE).
    """
    if len(models) < 2:
        return
    ref = models[reference_idx]
    others = [m for i, m in enumerate(models) if i != reference_idx]
    other = others[0]
    delta_col = f"delta_ece_reg__{other.label}__minus__{ref.label}"
    if delta_col not in compare_per_marker.columns:
        return
    ascending = direction == "improved"
    df = (
        compare_per_marker[["marker", delta_col]]
        .dropna()
        .sort_values(delta_col, ascending=ascending)
    )
    if df.empty:
        return
    top_df = df.head(top_n)
    top = top_df["marker"].tolist()

    cols = min(3, len(top))
    rows = ceil(len(top) / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.2 * cols, 3.0 * rows), squeeze=False, sharex=True, sharey=True
    )
    colors = _colors(len(models))
    for i, marker in enumerate(top):
        ax = axes[i // cols][i % cols]
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=0.8)
        for m, color in zip(models, colors):
            cov = m.coverage_curves[
                (m.coverage_curves["group_type"] == "per_marker")
                & (m.coverage_curves["group_value"] == marker)
            ].sort_values("alpha")
            g = m.global_metrics[
                (m.global_metrics["group_type"] == "per_marker")
                & (m.global_metrics["group_value"] == marker)
            ]
            ece = float(g["ece_reg"].iloc[0]) if len(g) else float("nan")
            ax.plot(
                cov["alpha"],
                cov["empirical_coverage"],
                color=color,
                linewidth=1.4,
                label=f"{m.label} (ECE={ece:.3f})",
            )
        delta_val = float(top_df.iloc[i][delta_col])
        ax.set_title(f"{marker}  (ΔECE={delta_val:+.3f})", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="both", labelsize=7)
        ax.legend(loc="lower right", fontsize=6)
    for j in range(len(top), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(
        f"{title_prefix} ({other.label} vs {ref.label})", fontsize=11
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_dir, stem)


def reliability_top_degraders(
    compare_per_marker: pd.DataFrame,
    models: list[ModelOutputs],
    out_dir: Path,
    reference_idx: int = 0,
    top_n: int = 6,
) -> None:
    """Reliability curves for the top-N markers where the non-reference model's
    per-marker ECE is largest above the reference (positive ΔECE)."""
    _reliability_top_n_grid(
        compare_per_marker,
        models,
        out_dir,
        reference_idx,
        top_n,
        direction="degraded",
        stem="reliability_top_degraders",
        title_prefix=f"Top-{top_n} degraders",
    )


def reliability_top_improvers(
    compare_per_marker: pd.DataFrame,
    models: list[ModelOutputs],
    out_dir: Path,
    reference_idx: int = 0,
    top_n: int = 6,
) -> None:
    """Reliability curves for the top-N markers where the non-reference model's
    per-marker ECE is largest below the reference (most-negative ΔECE)."""
    _reliability_top_n_grid(
        compare_per_marker,
        models,
        out_dir,
        reference_idx,
        top_n,
        direction="improved",
        stem="reliability_top_improvers",
        title_prefix=f"Top-{top_n} improvers",
    )


def coverage_gap_compare(models: list[ModelOutputs], out_dir: Path) -> None:
    """Per-α residual (empirical − nominal) overlaid for all models — exposes the
    sinusoidal coverage gap that the diagonal-axes reliability plot smooths over."""
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    colors = _colors(len(models))
    for m, color in zip(models, colors):
        cov = m.coverage_curves[m.coverage_curves["group_type"] == "global"].sort_values("alpha")
        g = m.global_metrics[m.global_metrics["group_type"] == "global"]
        ece = float(g["ece_reg"].iloc[0]) if len(g) else float("nan")
        ax.plot(
            cov["alpha"],
            cov["empirical_coverage"] - cov["alpha"],
            color=color,
            linewidth=2,
            label=f"{m.label}  (ECE={ece:.3f})",
        )
    ax.set_xlim(0, 1)
    ax.set_xlabel("nominal coverage α")
    ax.set_ylabel("empirical − nominal")
    ax.set_title("Coverage gap by α — global")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    _save(fig, out_dir, "coverage_gap_compare")


def auroc_aurc_per_marker_compare(
    compare_per_marker: pd.DataFrame,
    models: list[ModelOutputs],
    out_dir: Path,
    reference_idx: int = 0,
) -> None:
    """Two-panel paired bar chart: per-marker AUROC and AURC across models."""
    auroc_cols = [f"auroc_pixel_top10pct__{m.label}" for m in models]
    aurc_cols = [f"aurc_pixel__{m.label}" for m in models]
    if any(c not in compare_per_marker.columns for c in auroc_cols + aurc_cols):
        return
    df = compare_per_marker.dropna(subset=auroc_cols + aurc_cols).copy()
    if df.empty:
        return
    ref_label = models[reference_idx].label
    df = df.sort_values(f"auroc_pixel_top10pct__{ref_label}", ascending=True)

    colors = _colors(len(models))
    n_markers = len(df)
    bar_h = 0.8 / len(models)
    y = np.arange(n_markers)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, max(4.0, 0.25 * n_markers)), squeeze=False)
    for i, (m, color) in enumerate(zip(models, colors)):
        offset = (i - (len(models) - 1) / 2) * bar_h
        axes[0, 0].barh(
            y + offset, df[f"auroc_pixel_top10pct__{m.label}"],
            height=bar_h, color=color, label=m.label,
        )
        axes[0, 1].barh(
            y + offset, df[f"aurc_pixel__{m.label}"],
            height=bar_h, color=color, label=m.label,
        )
    axes[0, 0].set_yticks(y)
    axes[0, 0].set_yticklabels(df["marker"].tolist(), fontsize=7)
    axes[0, 0].axvline(0.5, color="grey", linestyle=":", linewidth=0.8, label="chance")
    axes[0, 0].set_xlabel("pixel AUROC (σ vs |r|>p₉₀)")
    axes[0, 0].set_title("Discrimination of large residuals")
    axes[0, 0].grid(axis="x", alpha=0.3)
    axes[0, 0].legend(loc="lower right", fontsize=7)

    axes[0, 1].set_yticks(y)
    axes[0, 1].set_yticklabels(df["marker"].tolist(), fontsize=7)
    axes[0, 1].set_xlabel("AURC")
    axes[0, 1].set_title("Risk-coverage area")
    axes[0, 1].grid(axis="x", alpha=0.3)
    axes[0, 1].legend(loc="lower right", fontsize=7)

    fig.suptitle("Per-marker pixel-uncertainty quality — paired bars", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, out_dir, "auroc_aurc_per_marker_compare")


def nll_per_marker_compare(
    compare_per_marker: pd.DataFrame,
    models: list[ModelOutputs],
    out_dir: Path,
    reference_idx: int = 0,
) -> None:
    """Paired bars of per-marker mean Gaussian NLL across models."""
    cols = [f"mean_nll__{m.label}" for m in models]
    if any(c not in compare_per_marker.columns for c in cols):
        return
    df = compare_per_marker.dropna(subset=cols).copy()
    if df.empty:
        return
    ref_label = models[reference_idx].label
    df = df.sort_values(f"mean_nll__{ref_label}", ascending=True)

    colors = _colors(len(models))
    n_markers = len(df)
    bar_h = 0.8 / len(models)
    y = np.arange(n_markers)

    fig, ax = plt.subplots(figsize=(7.0, max(4.0, 0.25 * n_markers)))
    for i, (m, color) in enumerate(zip(models, colors)):
        offset = (i - (len(models) - 1) / 2) * bar_h
        ax.barh(y + offset, df[f"mean_nll__{m.label}"], height=bar_h, color=color, label=m.label)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(df["marker"].tolist(), fontsize=7)
    ax.set_xlabel("mean Gaussian NLL")
    ax.set_title("Per-marker NLL — paired bars")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, out_dir, "nll_per_marker_compare")


def _risk_coverage_arrays(sigma: np.ndarray, sq: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(sigma)
    sq_sorted = sq[order]
    n = sq_sorted.size
    counts = np.arange(1, n + 1, dtype=np.float64)
    risk = np.cumsum(sq_sorted) / counts
    coverage = counts / n
    aurc_val = float(np.trapz(risk, coverage))
    return coverage, risk, aurc_val


def risk_coverage_global_compare(models: list[ModelOutputs], out_dir: Path) -> None:
    """Risk-coverage curves overlaid for all models, with each model's no-defer MSE.

    Loads each model's AUROC pool and computes the cumulative MSE on the
    lowest-σ fraction. Lower curves and lower AURC = σ is a better selective
    signal.
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    colors = _colors(len(models))
    for m, color in zip(models, colors):
        pool = load_pool(m)
        if pool is None or pool.empty:
            continue
        sigma = pool["sigma"].to_numpy(np.float64)
        abs_r = pool["abs_residual"].to_numpy(np.float64)
        sq = abs_r * abs_r
        coverage, risk, aurc_val = _risk_coverage_arrays(sigma, sq)
        overall = float(sq.mean())
        ax.plot(
            coverage, risk, color=color, linewidth=1.8,
            label=f"{m.label}  (AURC={aurc_val:.4f}, no-defer={overall:.4f})",
        )
        ax.axhline(overall, color=color, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("coverage (fraction of lowest-σ pixels kept)")
    ax.set_ylabel("MSE on retained pixels")
    ax.set_title("Risk-coverage — global")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "risk_coverage_global_compare")


def make_all_figures(
    models: list[ModelOutputs],
    markers: list[str],
    compare_per_marker: pd.DataFrame,
    f1_long: pd.DataFrame,
    paper_long: pd.DataFrame,
    out_dir: Path,
    reference_idx: int = 0,
) -> None:
    reliability_global_compare(models, out_dir)
    reliability_per_marker_compare(models, markers, out_dir)
    coverage_gap_compare(models, out_dir)
    ece_per_marker_compare(compare_per_marker, models, out_dir)
    ece_scatter_pairwise(compare_per_marker, models, out_dir, reference_idx)
    ece_delta_per_marker(compare_per_marker, models, out_dir, reference_idx)
    f1_per_marker_compare(compare_per_marker, models, out_dir)
    f1_global_vs_q_compare(f1_long, models, out_dir)
    sharpness_vs_ece_compare(compare_per_marker, models, out_dir, reference_idx)
    loo_scatter_compare(paper_long, models, out_dir)
    reliability_top_degraders(compare_per_marker, models, out_dir, reference_idx)
    reliability_top_improvers(compare_per_marker, models, out_dir, reference_idx)
    auroc_aurc_per_marker_compare(compare_per_marker, models, out_dir, reference_idx)
    nll_per_marker_compare(compare_per_marker, models, out_dir, reference_idx)
    risk_coverage_global_compare(models, out_dir)
