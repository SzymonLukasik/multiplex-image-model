"""All plotting functions read CSVs and write to figures/.

Each function takes paths only, never the upstream raw arrays.
"""
from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.metrics import roc_curve

from .auroc_subsample import aurc as _aurc_from_arrays


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def reliability_global(
    coverage_csv: Path, global_metrics_csv: Path, out_dir: Path, model_id: str
) -> None:
    cov = pd.read_csv(coverage_csv)
    glob = pd.read_csv(global_metrics_csv)
    cov_g = cov[cov["group_type"] == "global"].sort_values("alpha")
    glob_g = glob[glob["group_type"] == "global"]
    ece = float(glob_g["ece_reg"].iloc[0]) if len(glob_g) else float("nan")

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="ideal")
    ax.plot(
        cov_g["alpha"],
        cov_g["empirical_coverage"],
        color="C0",
        linewidth=2,
        label=f"model {model_id} (ECE={ece:.3f})",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("nominal coverage α")
    ax.set_ylabel("empirical coverage")
    ax.set_title(f"Reliability — global (LOO) — model {model_id}")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "reliability_global")


def reliability_per_marker_grid(
    coverage_csv: Path, global_metrics_csv: Path, out_dir: Path, model_id: str
) -> None:
    cov = pd.read_csv(coverage_csv)
    glob = pd.read_csv(global_metrics_csv)
    pm = glob[glob["group_type"] == "per_marker"].copy()
    if pm.empty:
        return
    pm = pm.sort_values("ece_reg", ascending=False)
    markers = pm["group_value"].tolist()

    n = len(markers)
    cols = min(6, max(3, int(np.ceil(np.sqrt(n)))))
    rows = ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.4 * rows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes).reshape(rows, cols)

    cov_pm = cov[cov["group_type"] == "per_marker"]

    for i, marker in enumerate(markers):
        ax = axes[i // cols, i % cols]
        sub = cov_pm[cov_pm["group_value"] == marker].sort_values("alpha")
        ece_m = float(pm[pm["group_value"] == marker]["ece_reg"].iloc[0])
        n_pix = int(pm[pm["group_value"] == marker]["n_pixels"].iloc[0])
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=0.8)
        ax.plot(sub["alpha"], sub["empirical_coverage"], color="C0", linewidth=1.4)
        ax.set_title(f"{marker}\nECE={ece_m:.3f}", fontsize=8)
        if n_pix < 1e5:
            ax.text(
                0.05,
                0.85,
                "low data",
                transform=ax.transAxes,
                fontsize=7,
                color="firebrick",
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="both", labelsize=7)
    # Hide empty axes
    for i in range(len(markers), rows * cols):
        axes[i // cols, i % cols].axis("off")

    fig.suptitle(f"Per-marker reliability (sorted by ECE↓) — model {model_id}", fontsize=11)
    fig.text(0.5, 0.04, "nominal coverage α", ha="center", fontsize=10)
    fig.text(0.04, 0.5, "empirical coverage", va="center", rotation="vertical", fontsize=10)
    fig.tight_layout(rect=[0.05, 0.05, 1, 0.97])
    _save(fig, out_dir, "reliability_per_marker_grid")


def paper_correlation_scatter(
    paper_csv: Path, global_metrics_csv: Path, out_dir: Path, model_id: str
) -> None:
    df = pd.read_csv(paper_csv)
    df = df[np.isfinite(df["log_mae_summary"]) & np.isfinite(df["log_var_summary"])]
    if df.empty:
        return
    glob = pd.read_csv(global_metrics_csv)
    g = glob[glob["group_type"] == "global"]
    pearson_r = float(g["pearson_logvar_logmae"].iloc[0]) if len(g) else float("nan")

    x = df["log_var_summary"].to_numpy()
    y = df["log_mae_summary"].to_numpy()

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(x, y, s=2, alpha=0.2, color="C0", linewidths=0)
    if x.size >= 2:
        lr = linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 200)
        ax.plot(
            xs,
            lr.intercept + lr.slope * xs,
            color="C3",
            linewidth=1.5,
            label=f"linear fit (slope={lr.slope:.2f})",
        )
    ax.set_xlabel(r"$\log\,\overline{\sigma^{2}}$ (per patch-channel)")
    ax.set_ylabel(r"$\log\,\mathrm{MAE}$ (per patch-channel)")
    ax.set_title(
        f"Model {model_id}\nPearson r = {pearson_r:.3f}"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    # ax.text(
    #     0.02,
    #     0.98,
    #     "computed under leave-one-out evaluation,\n"
    #     "a stricter regime than Figure 5's random-masking protocol.",
    #     transform=ax.transAxes,
    #     fontsize=7,
    #     color="grey",
    #     va="top",
    # )
    _save(fig, out_dir, "mae_variance_scatter")


def ece_per_marker(global_metrics_csv: Path, out_dir: Path, model_id: str) -> None:
    glob = pd.read_csv(global_metrics_csv)
    pm = glob[glob["group_type"] == "per_marker"].sort_values("ece_reg", ascending=True)
    if pm.empty:
        return
    glob_g = glob[glob["group_type"] == "global"]
    global_ece = float(glob_g["ece_reg"].iloc[0]) if len(glob_g) else float("nan")

    fig, ax = plt.subplots(figsize=(6.0, max(4.0, 0.22 * len(pm))))
    ax.barh(pm["group_value"], pm["ece_reg"], color="C0")
    ax.axvline(global_ece, color="C3", linestyle="--", label=f"global ECE={global_ece:.3f}")
    ax.set_xlabel("ECE_reg (mean |empirical − nominal|)")
    ax.set_title(f"Per-marker calibration error — model {model_id}")
    ax.legend(loc="lower right", fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="x", alpha=0.3)
    _save(fig, out_dir, "ece_per_marker")


def binary_mask_f1_per_marker(
    binary_csv: Path, out_dir: Path, model_id: str, q: float = 0.90
) -> None:
    bm = pd.read_csv(binary_csv)
    pm = bm[(bm["group_type"] == "per_marker") & (np.isclose(bm["quantile"], q))].copy()
    if pm.empty:
        return
    pm = pm.sort_values("f1", ascending=True)
    glob_q = bm[(bm["group_type"] == "global") & (np.isclose(bm["quantile"], q))]
    global_f1 = float(glob_q["f1"].iloc[0]) if len(glob_q) else float("nan")

    fig, ax = plt.subplots(figsize=(6.0, max(4.0, 0.22 * len(pm))))
    ax.barh(pm["group_value"], pm["f1"], color="C2")
    ax.axvline(global_f1, color="C3", linestyle="--", label=f"global F1={global_f1:.3f}")
    ax.set_xlabel(f"F1 (high-σ vs high-|r|, q={q})")
    ax.set_title(f"Per-marker binary-mask F1 — model {model_id}")
    ax.legend(loc="lower right", fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="x", alpha=0.3)
    _save(fig, out_dir, f"binary_mask_f1_per_marker_q{int(q*100)}")


def sharpness_vs_ece(global_metrics_csv: Path, out_dir: Path, model_id: str) -> None:
    glob = pd.read_csv(global_metrics_csv)
    pm = glob[glob["group_type"] == "per_marker"]
    if pm.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(pm["sharpness_mean_sigma"], pm["ece_reg"], color="C0", alpha=0.7)
    median_s = float(pm["sharpness_mean_sigma"].median())
    median_e = float(pm["ece_reg"].median())
    for _, row in pm.iterrows():
        in_corner = (
            row["sharpness_mean_sigma"] > median_s and row["ece_reg"] > median_e
        ) or (row["sharpness_mean_sigma"] < median_s and row["ece_reg"] < median_e)
        if in_corner:
            ax.annotate(
                str(row["group_value"]),
                xy=(row["sharpness_mean_sigma"], row["ece_reg"]),
                fontsize=7,
                xytext=(2, 2),
                textcoords="offset points",
                color="black",
            )
    ax.set_xlabel("sharpness (mean σ over pixels)")
    ax.set_ylabel("ECE_reg")
    ax.set_title(f"Sharpness vs ECE per marker — model {model_id}")
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "sharpness_vs_ece")


def _load_pool(csv_dir: Path) -> pd.DataFrame | None:
    parquet = csv_dir / "auroc_pool.parquet"
    csvgz = csv_dir / "auroc_pool.csv.gz"
    if parquet.exists():
        try:
            return pd.read_parquet(parquet)
        except Exception:
            pass
    if csvgz.exists():
        return pd.read_csv(csvgz)
    return None


def coverage_gap_curve(
    coverage_csv: Path, out_dir: Path, model_id: str
) -> None:
    """Per-α residual (empirical − nominal) for global + per-dataset curves.

    Highlights *where* on the α axis miscalibration sits: under-coverage
    (negative) vs over-coverage (positive).
    """
    cov = pd.read_csv(coverage_csv)
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    g = cov[cov["group_type"] == "global"].sort_values("alpha")
    ax.plot(
        g["alpha"],
        g["empirical_coverage"] - g["alpha"],
        color="C0",
        linewidth=2,
        label="global",
    )
    ds = cov[cov["group_type"] == "per_dataset"]
    for i, (name, sub) in enumerate(ds.groupby("group_value")):
        sub = sub.sort_values("alpha")
        ax.plot(
            sub["alpha"],
            sub["empirical_coverage"] - sub["alpha"],
            linewidth=1.0,
            alpha=0.8,
            label=f"dataset: {name}",
            color=f"C{(i + 1) % 10}",
        )
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("nominal coverage α")
    ax.set_ylabel("empirical − nominal")
    ax.set_title(f"Coverage gap by α — model {model_id}")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    _save(fig, out_dir, "coverage_gap_curve")


def auroc_aurc_per_marker(
    global_metrics_csv: Path, out_dir: Path, model_id: str
) -> None:
    """Two-panel bar chart: per-marker pixel AUROC@p90 (sort↓) and AURC (sort↑)."""
    glob = pd.read_csv(global_metrics_csv)
    pm = glob[glob["group_type"] == "per_marker"].copy()
    if pm.empty:
        return
    pm = pm.dropna(subset=["auroc_pixel_top10pct"])
    if pm.empty:
        return
    glob_g = glob[glob["group_type"] == "global"]
    g_auroc = float(glob_g["auroc_pixel_top10pct"].iloc[0]) if len(glob_g) else float("nan")
    g_aurc = float(glob_g["aurc_pixel"].iloc[0]) if len(glob_g) else float("nan")

    pm_a = pm.sort_values("auroc_pixel_top10pct", ascending=True)
    pm_r = pm.sort_values("aurc_pixel", ascending=False)

    fig, axes = plt.subplots(
        1, 2, figsize=(11.0, max(4.0, 0.22 * len(pm))), sharey=False
    )
    axes[0].barh(pm_a["group_value"], pm_a["auroc_pixel_top10pct"], color="C0")
    axes[0].axvline(g_auroc, color="C3", linestyle="--", label=f"global AUROC={g_auroc:.3f}")
    axes[0].axvline(0.5, color="grey", linestyle=":", linewidth=0.8, label="chance")
    axes[0].set_xlabel("pixel AUROC (σ vs |r|>p₉₀)")
    axes[0].set_title("Discrimination of large residuals")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].tick_params(axis="y", labelsize=7)
    axes[0].grid(axis="x", alpha=0.3)

    axes[1].barh(pm_r["group_value"], pm_r["aurc_pixel"], color="C2")
    axes[1].axvline(g_aurc, color="C3", linestyle="--", label=f"global AURC={g_aurc:.4f}")
    axes[1].set_xlabel("AURC")
    axes[1].set_title("Risk-coverage area")
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].tick_params(axis="y", labelsize=7)
    axes[1].grid(axis="x", alpha=0.3)

    fig.suptitle(f"Pixel-level uncertainty quality per marker — model {model_id}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, out_dir, "auroc_aurc_per_marker")


def nll_per_marker(global_metrics_csv: Path, out_dir: Path, model_id: str) -> None:
    glob = pd.read_csv(global_metrics_csv)
    pm = glob[glob["group_type"] == "per_marker"].copy()
    if pm.empty:
        return
    pm = pm.sort_values("mean_nll", ascending=True)
    glob_g = glob[glob["group_type"] == "global"]
    g_nll = float(glob_g["mean_nll"].iloc[0]) if len(glob_g) else float("nan")

    fig, ax = plt.subplots(figsize=(6.0, max(4.0, 0.22 * len(pm))))
    colours = ["C0" if v <= g_nll else "C3" for v in pm["mean_nll"]]
    ax.barh(pm["group_value"], pm["mean_nll"], color=colours)
    ax.axvline(g_nll, color="black", linestyle="--", label=f"global NLL={g_nll:.3f}")
    ax.set_xlabel("mean Gaussian NLL")
    ax.set_title(f"Per-marker NLL — model {model_id}")
    ax.legend(loc="lower right", fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="x", alpha=0.3)
    _save(fig, out_dir, "nll_per_marker")


def roc_curves_global(csv_dir: Path, out_dir: Path, model_id: str) -> None:
    """Pixel ROC curves (σ → |r|>p₉₀) for the global pool and a few datasets."""
    pool = _load_pool(csv_dir)
    if pool is None or pool.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="chance")

    def _plot_one(sub: pd.DataFrame, label: str, color: str) -> None:
        if len(sub) < 5_000:
            return
        abs_r = sub["abs_residual"].to_numpy()
        sigma = sub["sigma"].to_numpy()
        p90 = float(np.quantile(abs_r, 0.90))
        y = (abs_r > p90).astype(np.int8)
        if y.sum() == 0 or y.sum() == y.size:
            return
        fpr, tpr, _ = roc_curve(y, sigma)
        ax.plot(fpr, tpr, linewidth=1.6, label=label, color=color)

    _plot_one(pool, "global", "C0")
    for i, (name, sub) in enumerate(pool.groupby("dataset")):
        _plot_one(sub, f"dataset: {name}", f"C{(i + 1) % 10}")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"Pixel ROC: σ → |r|>p₉₀ — model {model_id}")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "roc_curves_global")


def risk_coverage_global(csv_dir: Path, out_dir: Path, model_id: str) -> None:
    """Risk-coverage curve from the AUROC pool: cumulative MSE vs coverage."""
    pool = _load_pool(csv_dir)
    if pool is None or pool.empty:
        return
    sigma = pool["sigma"].to_numpy(np.float64)
    abs_r = pool["abs_residual"].to_numpy(np.float64)
    sq = abs_r * abs_r
    order = np.argsort(sigma)
    sq_sorted = sq[order]
    counts = np.arange(1, sq_sorted.size + 1, dtype=np.float64)
    risk = np.cumsum(sq_sorted) / counts
    coverage = counts / sq_sorted.size
    aurc_val = _aurc_from_arrays(sigma, sq)
    overall = float(sq.mean())

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(coverage, risk, color="C0", linewidth=1.6, label=f"selective (AURC={aurc_val:.4f})")
    ax.axhline(overall, color="grey", linestyle="--", linewidth=1, label=f"no-defer MSE={overall:.4f}")
    ax.set_xlabel("coverage (fraction of lowest-σ pixels kept)")
    ax.set_ylabel("MSE on retained pixels")
    ax.set_title(f"Risk–coverage — model {model_id}")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "risk_coverage_global")


def _per_marker_panel_grid(
    n_markers: int, panel_size: float = 2.4, sharey: bool = True
) -> tuple[plt.Figure, np.ndarray, int, int]:
    """Build a small-multiples grid sized for n_markers."""
    cols = min(7, max(3, int(np.ceil(np.sqrt(n_markers)))))
    rows = ceil(n_markers / cols)
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(panel_size * cols, panel_size * rows),
        sharex=True, sharey=sharey,
    )
    axes = np.atleast_2d(axes).reshape(rows, cols)
    return fig, axes, rows, cols


def roc_curves_per_marker(
    csv_dir: Path,
    global_metrics_csv: Path,
    out_dir: Path,
    model_id: str,
    min_pixels: int = 5_000,
) -> None:
    """Per-marker pixel ROC (σ → |r|>p₉₀), small-multiples grid sorted by AUROC asc."""
    pool = _load_pool(csv_dir)
    if pool is None or pool.empty or "marker" not in pool.columns:
        return
    glob = pd.read_csv(global_metrics_csv)
    pm = glob[glob["group_type"] == "per_marker"].copy()
    pm = pm.dropna(subset=["auroc_pixel_top10pct"])
    if pm.empty:
        return
    pm = pm.sort_values("auroc_pixel_top10pct", ascending=True)
    markers = pm["group_value"].astype(str).tolist()
    auroc_lookup = dict(zip(pm["group_value"].astype(str), pm["auroc_pixel_top10pct"]))

    pool_by_marker = {str(k): v for k, v in pool.groupby("marker")}
    drawable = [m for m in markers if len(pool_by_marker.get(m, [])) >= min_pixels]
    if not drawable:
        return

    fig, axes, rows, cols = _per_marker_panel_grid(len(drawable))
    for i, marker in enumerate(drawable):
        ax = axes[i // cols, i % cols]
        sub = pool_by_marker[marker]
        abs_r = sub["abs_residual"].to_numpy()
        sigma = sub["sigma"].to_numpy()
        p90 = float(np.quantile(abs_r, 0.90))
        y = (abs_r > p90).astype(np.int8)
        if y.sum() == 0 or y.sum() == y.size:
            ax.axis("off")
            continue
        fpr, tpr, _ = roc_curve(y, sigma)
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=0.7)
        ax.plot(fpr, tpr, color="C0", linewidth=1.3)
        ax.set_title(f"{marker}\nAUROC={auroc_lookup[marker]:.3f}", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="both", labelsize=7)

    for j in range(len(drawable), rows * cols):
        axes[j // cols, j % cols].axis("off")

    fig.suptitle(
        f"Per-marker pixel ROC: σ → |r|>p₉₀ (sorted by AUROC↑) — model {model_id}",
        fontsize=11,
    )
    fig.text(0.5, 0.04, "FPR", ha="center", fontsize=10)
    fig.text(0.04, 0.5, "TPR", va="center", rotation="vertical", fontsize=10)
    fig.tight_layout(rect=[0.05, 0.05, 1, 0.97])
    _save(fig, out_dir, "roc_curves_per_marker")


def risk_coverage_per_marker(
    csv_dir: Path,
    global_metrics_csv: Path,
    out_dir: Path,
    model_id: str,
    min_pixels: int = 5_000,
) -> None:
    """Per-marker risk-coverage curves, small-multiples grid sorted by AURC desc.

    Each panel: x=coverage, y=mean MSE on retained (lowest-σ) pixels; horizontal
    dashed reference at the marker's no-defer mean MSE. Y-axis is per-panel
    because absolute MSE varies by ~100× across markers.
    """
    pool = _load_pool(csv_dir)
    if pool is None or pool.empty or "marker" not in pool.columns:
        return
    glob = pd.read_csv(global_metrics_csv)
    pm = glob[glob["group_type"] == "per_marker"].copy()
    pm = pm.dropna(subset=["aurc_pixel"])
    if pm.empty:
        return
    pm = pm.sort_values("aurc_pixel", ascending=False)
    markers = pm["group_value"].astype(str).tolist()
    aurc_lookup = dict(zip(pm["group_value"].astype(str), pm["aurc_pixel"]))

    pool_by_marker = {str(k): v for k, v in pool.groupby("marker")}
    drawable = [m for m in markers if len(pool_by_marker.get(m, [])) >= min_pixels]
    if not drawable:
        return

    fig, axes, rows, cols = _per_marker_panel_grid(
        len(drawable), panel_size=2.6, sharey=False
    )

    for i, marker in enumerate(drawable):
        ax = axes[i // cols, i % cols]
        sub = pool_by_marker[marker]
        sigma = sub["sigma"].to_numpy(np.float64)
        abs_r = sub["abs_residual"].to_numpy(np.float64)
        sq = abs_r * abs_r
        order = np.argsort(sigma)
        sq_sorted = sq[order]
        n = sq_sorted.size
        counts = np.arange(1, n + 1, dtype=np.float64)
        risk = np.cumsum(sq_sorted) / counts
        coverage = counts / n
        overall = float(sq.mean())
        ax.plot(coverage, risk, color="C0", linewidth=1.3)
        ax.axhline(overall, color="grey", linestyle="--", linewidth=0.7)
        ax.set_title(f"{marker}\nAURC={aurc_lookup[marker]:.4f}", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(overall, float(risk.max())) * 1.05)
        ax.tick_params(axis="both", labelsize=7)

    for j in range(len(drawable), rows * cols):
        axes[j // cols, j % cols].axis("off")

    fig.suptitle(
        f"Per-marker risk–coverage (sorted by AURC↓) — model {model_id}",
        fontsize=11,
    )
    fig.text(0.5, 0.04, "coverage (lowest-σ kept)", ha="center", fontsize=10)
    fig.text(
        0.04, 0.5, "MSE on retained pixels", va="center", rotation="vertical", fontsize=10
    )
    fig.tight_layout(rect=[0.05, 0.05, 1, 0.97])
    _save(fig, out_dir, "risk_coverage_per_marker")


def sigma_residual_density(csv_dir: Path, out_dir: Path, model_id: str) -> None:
    """2D log-density of (σ, |r|) on the global pool, with y=x reference."""
    pool = _load_pool(csv_dir)
    if pool is None or pool.empty:
        return
    sigma = pool["sigma"].to_numpy(np.float64)
    abs_r = pool["abs_residual"].to_numpy(np.float64)
    eps = 1e-9
    x = np.log10(sigma + eps)
    y = np.log10(abs_r + eps)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        return

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    h = ax.hexbin(x, y, gridsize=80, cmap="viridis", bins="log", mincnt=1)
    fig.colorbar(h, ax=ax, label="log10(count)")
    lim_lo = float(min(x.min(), y.min()))
    lim_hi = float(max(x.max(), y.max()))
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="white", linestyle="--", linewidth=1)
    ax.set_xlabel(r"$\log_{10}\,\sigma$")
    ax.set_ylabel(r"$\log_{10}\,|r|$")
    ax.set_title(f"Pixel σ vs residual density — model {model_id}")
    _save(fig, out_dir, "sigma_residual_density")


def metric_summary_heatmap(
    global_metrics_csv: Path, out_dir: Path, model_id: str
) -> None:
    """Per-marker normalised heatmap of headline metrics (z-scored across markers)."""
    glob = pd.read_csv(global_metrics_csv)
    pm = glob[glob["group_type"] == "per_marker"].copy()
    if pm.empty:
        return
    cols = [
        "ece_reg",
        "mean_nll",
        "sharpness_mean_sigma",
        "rmse",
        "mae",
        "auroc_pixel_top10pct",
        "aurc_pixel",
        "pearson_logvar_logmae",
    ]
    cols = [c for c in cols if c in pm.columns]
    pm = pm.sort_values("ece_reg", ascending=True)
    M = pm[cols].to_numpy(dtype=np.float64)
    # z-score per column; flip sign for "higher-is-better" metrics so that
    # warm = worse on every column.
    higher_is_better = {"auroc_pixel_top10pct", "pearson_logvar_logmae"}
    Z = np.empty_like(M)
    for j, c in enumerate(cols):
        col = M[:, j]
        mu = np.nanmean(col)
        sd = np.nanstd(col)
        z = (col - mu) / sd if sd > 0 else np.zeros_like(col)
        if c in higher_is_better:
            z = -z
        Z[:, j] = z

    fig, ax = plt.subplots(figsize=(0.55 * len(cols) + 3.5, max(4.0, 0.22 * len(pm))))
    im = ax.imshow(Z, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_yticks(range(len(pm)))
    ax.set_yticklabels(pm["group_value"].tolist(), fontsize=7)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=8)
    ax.set_title(f"Per-marker metric z-scores (warm = worse) — model {model_id}")
    fig.colorbar(im, ax=ax, label="z-score (worse →)")
    _save(fig, out_dir, "metric_summary_heatmap")


def make_all_figures(out_dir: Path, model_id: str) -> None:
    csv_dir = out_dir / "csv"
    fig_dir = out_dir / "figures"

    coverage_csv = csv_dir / "coverage_curves.csv"
    global_csv = csv_dir / "global_metrics.csv"
    paper_csv = csv_dir / "paper_correlation.csv"
    binary_csv = csv_dir / "binary_mask_analysis.csv"

    if coverage_csv.exists() and global_csv.exists():
        reliability_global(coverage_csv, global_csv, fig_dir, model_id)
        reliability_per_marker_grid(coverage_csv, global_csv, fig_dir, model_id)
    if coverage_csv.exists():
        coverage_gap_curve(coverage_csv, fig_dir, model_id)
    if global_csv.exists():
        ece_per_marker(global_csv, fig_dir, model_id)
        sharpness_vs_ece(global_csv, fig_dir, model_id)
        auroc_aurc_per_marker(global_csv, fig_dir, model_id)
        nll_per_marker(global_csv, fig_dir, model_id)
        metric_summary_heatmap(global_csv, fig_dir, model_id)
    if paper_csv.exists() and global_csv.exists():
        paper_correlation_scatter(paper_csv, global_csv, fig_dir, model_id)
    if binary_csv.exists():
        for q in (0.90, 0.95, 0.99):
            binary_mask_f1_per_marker(binary_csv, fig_dir, model_id, q=q)
    # Pool-backed figures (require AUROC pool on disk).
    roc_curves_global(csv_dir, fig_dir, model_id)
    risk_coverage_global(csv_dir, fig_dir, model_id)
    sigma_residual_density(csv_dir, fig_dir, model_id)
    if global_csv.exists():
        roc_curves_per_marker(csv_dir, global_csv, fig_dir, model_id)
        risk_coverage_per_marker(csv_dir, global_csv, fig_dir, model_id)
