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
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
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
    """NeurIPS-ready reliability diagram with a coverage-gap inset.

    Main panel keeps the canonical y=x reliability view so readers see
    that all curves track the diagonal. The inset shows the residual
    (empirical − nominal) on a magnified y-axis where the per-model
    differences — invisible at unit-square scale — become legible.
    """
    from matplotlib.ticker import MultipleLocator

    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    colors = _colors(len(models))

    ax.plot(
        [0, 1], [0, 1],
        color="grey", linestyle="--", linewidth=0.9, zorder=1,
        label=r"perfect calibration ($y=\alpha$)",
    )
    for m, color in zip(models, colors):
        cov = (
            m.coverage_curves[m.coverage_curves["group_type"] == "global"]
            .sort_values("alpha")
        )
        g = m.global_metrics[m.global_metrics["group_type"] == "global"]
        ece = float(g["ece_reg"].iloc[0]) if len(g) else float("nan")
        ax.plot(
            cov["alpha"], cov["empirical_coverage"],
            color=color, linewidth=1.6, zorder=3,
            label=f"{m.label}  (ECE={ece:.3f})",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"nominal coverage $\alpha$", fontsize=10)
    ax.set_ylabel("empirical coverage", fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="both", which="major", labelsize=8, length=3)
    ax.tick_params(axis="both", which="minor", length=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.grid(which="major", alpha=0.20, linewidth=0.5)
    ax.grid(which="minor", alpha=0.08, linewidth=0.4)
    ax.legend(
        loc="upper left", fontsize=7.5, frameon=False,
        handlelength=1.6, borderaxespad=0.4,
    )

    # ---- inset: coverage-gap residual on a magnified y-axis ----
    ax_in = ax.inset_axes([0.56, 0.08, 0.40, 0.34])
    ax_in.axhline(0.0, color="grey", linestyle="--", linewidth=0.7, zorder=1)
    gap_max = 0.0
    for m, color in zip(models, colors):
        cov = (
            m.coverage_curves[m.coverage_curves["group_type"] == "global"]
            .sort_values("alpha")
        )
        gap = cov["empirical_coverage"].to_numpy() - cov["alpha"].to_numpy()
        gap_max = max(gap_max, float(np.max(np.abs(gap))))
        ax_in.plot(cov["alpha"], gap, color=color, linewidth=1.2, zorder=3)

    y_lim = max(0.02, np.ceil(gap_max * 100) / 100)
    ax_in.set_xlim(0, 1)
    ax_in.set_ylim(-y_lim, y_lim)
    ax_in.set_title("empirical $-$ nominal", fontsize=7.5, pad=2)
    ax_in.xaxis.set_major_locator(MultipleLocator(0.5))
    ax_in.yaxis.set_major_locator(MultipleLocator(y_lim))
    ax_in.tick_params(axis="both", labelsize=6.5, length=2, pad=1)
    for side in ("top", "right"):
        ax_in.spines[side].set_visible(False)
    ax_in.spines["left"].set_linewidth(0.5)
    ax_in.spines["bottom"].set_linewidth(0.5)
    ax_in.grid(alpha=0.15, linewidth=0.4)
    ax_in.patch.set_alpha(0.92)

    fig.tight_layout()
    _save(fig, out_dir, "reliability_global_compare")


def reliability_global_compare_paper(
    models: list[ModelOutputs], out_dir: Path
) -> None:
    """NeurIPS-ready reliability diagram: clean spines, two-panel view
    (reliability curve + coverage-gap residual) so the deviation from y=x
    is unmistakable.
    """
    from matplotlib.ticker import MultipleLocator

    fig, (ax_rel, ax_gap) = plt.subplots(
        1, 2,
        figsize=(8.6, 3.8),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.28},
    )
    colors = _colors(len(models))

    # ---------- Panel 1: reliability curve ----------
    ax_rel.plot(
        [0, 1], [0, 1],
        color="grey", linestyle="--", linewidth=0.9, label="ideal", zorder=1,
    )
    for m, color in zip(models, colors):
        cov = (
            m.coverage_curves[m.coverage_curves["group_type"] == "global"]
            .sort_values("alpha")
        )
        g = m.global_metrics[m.global_metrics["group_type"] == "global"]
        ece = float(g["ece_reg"].iloc[0]) if len(g) else float("nan")
        ax_rel.plot(
            cov["alpha"], cov["empirical_coverage"],
            color=color, linewidth=1.6, zorder=3,
            label=f"{m.label}  (ECE={ece:.3f})",
        )

    ax_rel.set_xlim(0, 1)
    ax_rel.set_ylim(0, 1)
    ax_rel.set_xlabel(r"nominal coverage $\alpha$", fontsize=10)
    ax_rel.set_ylabel("empirical coverage", fontsize=10)
    ax_rel.set_aspect("equal", adjustable="box")
    ax_rel.xaxis.set_major_locator(MultipleLocator(0.2))
    ax_rel.yaxis.set_major_locator(MultipleLocator(0.2))
    ax_rel.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax_rel.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax_rel.tick_params(axis="both", which="major", labelsize=8, length=3)
    ax_rel.tick_params(axis="both", which="minor", length=2)
    for side in ("top", "right"):
        ax_rel.spines[side].set_visible(False)
    ax_rel.spines["left"].set_linewidth(0.6)
    ax_rel.spines["bottom"].set_linewidth(0.6)
    ax_rel.grid(which="major", alpha=0.20, linewidth=0.5)
    ax_rel.grid(which="minor", alpha=0.08, linewidth=0.4)
    ax_rel.legend(loc="lower right", fontsize=8, frameon=False, handlelength=1.6)

    # ---------- Panel 2: coverage-gap residual ----------
    ax_gap.axhline(0.0, color="grey", linestyle="--", linewidth=0.9, zorder=1)
    for m, color in zip(models, colors):
        cov = (
            m.coverage_curves[m.coverage_curves["group_type"] == "global"]
            .sort_values("alpha")
        )
        ax_gap.plot(
            cov["alpha"], cov["empirical_coverage"] - cov["alpha"],
            color=color, linewidth=1.6, zorder=3, label=m.label,
        )

    ax_gap.set_xlim(0, 1)
    ax_gap.set_xlabel(r"nominal coverage $\alpha$", fontsize=10)
    ax_gap.set_ylabel("empirical $-$ nominal", fontsize=10)
    ax_gap.xaxis.set_major_locator(MultipleLocator(0.2))
    ax_gap.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax_gap.yaxis.set_major_locator(MultipleLocator(0.02))
    ax_gap.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax_gap.tick_params(axis="both", which="major", labelsize=8, length=3)
    ax_gap.tick_params(axis="both", which="minor", length=2)
    for side in ("top", "right"):
        ax_gap.spines[side].set_visible(False)
    ax_gap.spines["left"].set_linewidth(0.6)
    ax_gap.spines["bottom"].set_linewidth(0.6)
    ax_gap.grid(which="major", alpha=0.20, linewidth=0.5)
    ax_gap.grid(which="minor", alpha=0.08, linewidth=0.4)

    fig.tight_layout()
    _save(fig, out_dir, "reliability_global_compare_paper")


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
        ece_lines: list[tuple[str, float, str]] = []
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
            ece_lines.append((m.label, ece, color))
        # Marker name as the title (always fits inside the panel).
        ax.set_title(marker, fontsize=8)
        # Per-model ECE values as a stacked, color-coded text block in the
        # upper-left corner; this never spills into the next column regardless
        # of N or label length.
        for k, (lab, val, color) in enumerate(ece_lines):
            # Don't truncate mathtext labels — slicing $...$ mid-expression
            # produces a parse error.
            is_mathtext = "$" in lab
            if is_mathtext or len(lab) <= 10:
                short = lab
            else:
                short = f"{lab[:9]}…"
            ax.text(
                0.04,
                0.96 - 0.10 * k,
                f"{short}: {val:.3f}",
                transform=ax.transAxes,
                fontsize=6.5,
                color=color,
                va="top",
                ha="left",
            )
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
        present = [
            (m, color)
            for m, color in zip(models, colors)
            if f"f1_{qtag}__{m.label}" in compare_per_marker.columns
        ]
        missing = [
            m.label for m in models
            if f"f1_{qtag}__{m.label}" not in compare_per_marker.columns
        ]
        if not present:
            ax.text(0.5, 0.5,
                    f"q={q}: no F1 data for any model\n(missing AUROC-pool dataset column?)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="firebrick")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        cols = [f"f1_{qtag}__{m.label}" for m, _ in present]
        df = compare_per_marker.dropna(subset=cols, how="all").copy()
        if df.empty:
            ax.text(0.5, 0.5, f"q={q}: all F1 values are NaN",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="firebrick")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        df = df.sort_values(cols[0], ascending=True)
        n_markers = len(df)
        bar_h = 0.8 / max(len(present), 1)
        y = np.arange(n_markers)
        for i, (m, color) in enumerate(present):
            offset = (i - (len(present) - 1) / 2) * bar_h
            ax.barh(y + offset, df[f"f1_{qtag}__{m.label}"],
                    height=bar_h, color=color, label=m.label)

        # x-limit: cap a hair above max F1; always start at 0 so the chance
        # line and the bars share an honest origin.
        x_max = float(df[cols].max(numeric_only=True).max())
        x_max = max(x_max * 1.05, 1.0 - q + 0.02)
        ax.set_xlim(0.0, x_max)

        # Tick layout that always shows the chance value (1-q) explicitly,
        # plus a clean major grid every 0.1 and minor ticks at 0.05.
        chance = 1.0 - q
        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        ax.tick_params(axis="x", which="major", length=4, labelsize=8)
        ax.tick_params(axis="x", which="minor", length=2)
        # Add an extra labelled tick at the chance value, keeping the
        # auto-generated ones — guarantees q=0.95 (chance=0.05) and
        # q=0.99 (chance=0.01) are readable from the axis.
        existing = list(ax.get_xticks())
        if not any(abs(t - chance) < 1e-3 for t in existing):
            ax.set_xticks(sorted(existing + [chance]))

        # Vertical chance line + in-plot annotation (so you don't have to
        # read it off the axis under tight margins).
        ax.axvline(chance, color="black", linestyle="--", linewidth=0.8)
        ax.annotate(
            f"chance = {chance:.2f}",
            xy=(chance, n_markers - 0.5),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=7,
            va="top",
            color="black",
        )

        ax.set_yticks(y)
        ax.set_yticklabels(df["marker"].tolist(), fontsize=7)
        ax.set_xlabel(f"F1 (q={q})")
        title = f"q = {q}"
        if missing:
            title += f"\n(missing: {', '.join(missing)})"
        ax.set_title(title, fontsize=9)

        # NeurIPS-clean: drop top + right spines, keep a faint x-grid only.
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.grid(axis="x", which="major", alpha=0.25, linewidth=0.5)
        ax.grid(axis="x", which="minor", alpha=0.10, linewidth=0.4)

        ax.legend(loc="lower right", fontsize=7, frameon=False)
    fig.suptitle("Per-marker σ-vs-|residual| F1", fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir, "f1_per_marker_compare")


def f1_global_vs_q_compare(
    f1_long: pd.DataFrame, models: list[ModelOutputs], out_dir: Path
) -> None:
    if f1_long.empty:
        return
    from matplotlib.ticker import MultipleLocator

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    colors = _colors(len(models))
    for m, color in zip(models, colors):
        sub = f1_long[f1_long["label"] == m.label].sort_values("quantile")
        ax.plot(sub["quantile"], sub["f1"], color=color, marker="o", label=m.label)

    qs_actual = sorted(f1_long["quantile"].unique().tolist())
    qs = np.linspace(f1_long["quantile"].min(), f1_long["quantile"].max(), 50)
    ax.plot(qs, 1.0 - qs, color="black", linestyle="--", linewidth=0.8, label="chance (1−q)")

    # Honest origin + tick at every measured q so the gap to the chance line
    # is unambiguous; rely on the dashed line + legend for the chance value.
    ax.set_xlim(min(qs_actual) - 0.005, max(qs_actual) + 0.005)
    ax.set_xticks(qs_actual)
    ax.set_xticklabels([f"{q:.2f}" for q in qs_actual])
    ax.set_ylim(bottom=0.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.tick_params(axis="y", which="major", length=4, labelsize=8)
    ax.tick_params(axis="y", which="minor", length=2)

    ax.set_xlabel("quantile")
    ax.set_ylabel("F1 (global, σ vs |r|)")
    # ax.set_title("Global F1 vs strictness")

    # NeurIPS-clean: drop top + right spines, thin remaining ones, faint grid.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.grid(axis="y", which="major", alpha=0.25, linewidth=0.5)
    ax.grid(axis="y", which="minor", alpha=0.10, linewidth=0.4)

    ax.legend(loc="upper right", fontsize=8, frameon=False)
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
    # ax.set_title("Sharpness vs ECE per marker")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "sharpness_vs_ece_compare")


def sharpness_vs_ece_compare_paper(
    compare_per_marker: pd.DataFrame,
    models: list[ModelOutputs],
    out_dir: Path,
) -> None:
    """NeurIPS-ready faceted version: one panel per model, every marker
    annotated. Shared axes so cross-model shifts are eye-ballable.
    """
    from matplotlib.ticker import MaxNLocator

    valid_models = [
        m for m in models
        if f"sharpness_mean_sigma__{m.label}" in compare_per_marker.columns
        and f"ece_reg__{m.label}" in compare_per_marker.columns
    ]
    if not valid_models:
        return

    # Compute shared axis limits across all models for honest visual comparison.
    x_all, y_all = [], []
    for m in valid_models:
        x_all.append(compare_per_marker[f"sharpness_mean_sigma__{m.label}"])
        y_all.append(compare_per_marker[f"ece_reg__{m.label}"])
    x_concat = pd.concat(x_all).dropna()
    y_concat = pd.concat(y_all).dropna()
    if x_concat.empty or y_concat.empty:
        return
    pad_x = (x_concat.max() - x_concat.min()) * 0.08 or 0.005
    pad_y = (y_concat.max() - y_concat.min()) * 0.10 or 0.005
    xlim = (max(0.0, x_concat.min() - pad_x), x_concat.max() + pad_x)
    ylim = (max(0.0, y_concat.min() - pad_y), y_concat.max() + pad_y)

    n = len(valid_models)
    fig, axes = plt.subplots(
        1, n,
        figsize=(3.6 * n, 3.6),
        sharex=True, sharey=True,
        squeeze=False,
    )
    colors = _colors(len(models))
    color_for = {m.label: c for m, c in zip(models, colors)}

    median_x = float(x_concat.median())
    median_y = float(y_concat.median())

    for ax, m in zip(axes[0], valid_models):
        x = compare_per_marker[f"sharpness_mean_sigma__{m.label}"]
        y = compare_per_marker[f"ece_reg__{m.label}"]
        markers = compare_per_marker["marker"]
        valid = x.notna() & y.notna()
        ax.scatter(
            x[valid], y[valid],
            s=22, alpha=0.85, color=color_for[m.label],
            edgecolors="white", linewidths=0.6,
            zorder=3,
        )

        # Annotate every marker. Use small fontsize and a small offset so
        # labels are readable but not overwhelming.
        for xi, yi, lab in zip(x[valid], y[valid], markers[valid]):
            ax.annotate(
                str(lab),
                xy=(float(xi), float(yi)),
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=6,
                color="#222222",
                zorder=4,
            )

        # Subtle median guides — separates "high-σ high-ECE" markers from
        # "low-σ low-ECE" without dominating the panel.
        ax.axvline(median_x, color="grey", linestyle=":", linewidth=0.6, alpha=0.6)
        ax.axhline(median_y, color="grey", linestyle=":", linewidth=0.6, alpha=0.6)

        ax.set_title(m.label, fontsize=10, pad=6)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
        ax.tick_params(axis="both", labelsize=8, length=3)

        # NeurIPS-clean spines + faint grid.
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.grid(alpha=0.18, linewidth=0.4, zorder=0)

    # Single shared axis labels along the bottom-left, no per-panel duplication.
    axes[0, 0].set_ylabel("ECE", fontsize=10)
    for ax in axes[0]:
        ax.set_xlabel(r"sharpness  $\overline{\hat\sigma}$", fontsize=10)

    fig.tight_layout()
    _save(fig, out_dir, "sharpness_vs_ece_compare_paper")


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
    reliability_global_compare_paper(models, out_dir)
    reliability_per_marker_compare(models, markers, out_dir)
    coverage_gap_compare(models, out_dir)
    ece_per_marker_compare(compare_per_marker, models, out_dir)
    ece_scatter_pairwise(compare_per_marker, models, out_dir, reference_idx)
    ece_delta_per_marker(compare_per_marker, models, out_dir, reference_idx)
    f1_per_marker_compare(compare_per_marker, models, out_dir)
    f1_global_vs_q_compare(f1_long, models, out_dir)
    sharpness_vs_ece_compare(compare_per_marker, models, out_dir, reference_idx)
    sharpness_vs_ece_compare_paper(compare_per_marker, models, out_dir)
    loo_scatter_compare(paper_long, models, out_dir)
    reliability_top_degraders(compare_per_marker, models, out_dir, reference_idx)
    reliability_top_improvers(compare_per_marker, models, out_dir, reference_idx)
    auroc_aurc_per_marker_compare(compare_per_marker, models, out_dir, reference_idx)
    nll_per_marker_compare(compare_per_marker, models, out_dir, reference_idx)
    risk_coverage_global_compare(models, out_dir)
