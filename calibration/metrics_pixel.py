"""Per-(patch, channel) pixel-level metrics.

A worker function processes a list of npz paths and returns
  (rows: list[dict], pool_chunk: pd.DataFrame)
where `rows` is one row per (patch, channel) and `pool_chunk` is the
subsampled-pixel pool for AUROC/AURC.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from .auroc_subsample import subsample_pc
from .config import (
    ALPHA_GRID,
    BINARY_MASK_QUANTILES,
    DEFAULT_PER_PC_CAP,
    VAR_CLIP,
    Z_ALPHA_GRID,
    alpha_col,
    quantile_cols,
)
from .io_utils import parse_metadata

LOG2PI = float(np.log(2.0 * np.pi))


def _patch_channel_row(
    target: np.ndarray,
    recon: np.ndarray,
    var: np.ndarray,
    sigma: np.ndarray,
    marker: str,
    channel_id: int,
    meta: dict,
    npz_path: str,
) -> dict:
    r = (target - recon).astype(np.float64, copy=False)
    abs_r = np.abs(r)
    sq = r * r
    n = r.size

    # Base summaries
    mse = float(np.mean(sq))
    mae = float(np.mean(abs_r))
    rmse = float(np.sqrt(mse))
    mean_var = float(np.mean(var))
    mean_sigma = float(np.mean(sigma))
    mean_logvar = float(np.mean(np.log(var)))

    # NLL (Gaussian)
    # 0.5 * (log(2π) + log σ² + r²/σ²)
    nll = 0.5 * (LOG2PI + np.log(var) + sq / var)
    mean_nll = float(np.mean(nll))

    log_mae_summary = float(np.log(mae)) if mae > 0 else float("-inf")
    log_var_summary = float(np.log(mean_var))

    row: dict = {
        "npz_path": npz_path,
        "image_index": int(meta.get("image_index", -1)),
        "image_path": str(meta.get("image_path", "")),
        "dataset_name": str(meta.get("dataset_name", "")),
        "marker_name": marker,
        "channel_id": int(channel_id),
        "n_pixels": int(n),
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "mean_var": mean_var,
        "mean_sigma": mean_sigma,
        "mean_logvar": mean_logvar,
        "mean_nll": mean_nll,
        "log_mae_summary": log_mae_summary,
        "log_var_summary": log_var_summary,
    }

    # Coverage counts at each α (vectorised over α to keep this fast).
    # in_α(pixel) := (|r| <= z_α * σ); we want the count over pixels
    # for each α. Compute |r|/σ once, then compare against z_α.
    ratio = abs_r / sigma  # finite because var has been clipped
    # broadcast: (n,) >= (n_alpha,) -> use sort-and-search for efficiency.
    ratio_sorted = np.sort(ratio)
    # cov_count[a] = number of ratios <= z_α[a]
    cov_counts = np.searchsorted(ratio_sorted, Z_ALPHA_GRID, side="right")
    for a, c in zip(ALPHA_GRID, cov_counts):
        row[alpha_col(a)] = int(c)

    # Within-(patch, channel) binary-mask analysis at fixed quantiles.
    # Note: group-level analysis recomputes thresholds from a global pool;
    # this CSV column captures per-pc agreement.
    abs_r_sorted = np.sort(abs_r)
    sigma_sorted = np.sort(sigma)
    for q in BINARY_MASK_QUANTILES:
        sigma_thr = float(np.quantile(sigma_sorted, q))
        r_thr = float(np.quantile(abs_r_sorted, q))
        m_sigma = sigma > sigma_thr
        m_r = abs_r > r_thr
        tp = int(np.sum(m_sigma & m_r))
        fp = int(np.sum(m_sigma & ~m_r))
        fn = int(np.sum(~m_sigma & m_r))
        tn = int(n - tp - fp - fn)
        cols = quantile_cols(q)
        row[cols["tp"]] = tp
        row[cols["fp"]] = fp
        row[cols["fn"]] = fn
        row[cols["tn"]] = tn
        row[cols["sigma_thr"]] = sigma_thr
        row[cols["r_thr"]] = r_thr

    return row


def process_npz_chunk(
    npz_paths: list[str],
    seed: int,
    per_pc_cap: int = DEFAULT_PER_PC_CAP,
) -> tuple[list[dict], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    pool_sigma: list[np.ndarray] = []
    pool_absr: list[np.ndarray] = []
    pool_marker: list[np.ndarray] = []
    pool_dataset: list[np.ndarray] = []

    for npz_path in npz_paths:
        with np.load(npz_path, allow_pickle=True) as d:
            meta = parse_metadata(d["metadata"])
            channel_ids = d["channel_ids"]
            masked_ids = d["masked_channel_ids"]
            if set(channel_ids.tolist()) != set(masked_ids.tolist()):
                # Skip — this file does not match the LOO assumption.
                # Caller can detect via missing rows; we record nothing.
                continue
            recon = d["recon"]
            target = d["target"]
            variance = d["variance"]
            marker_names = d["marker_names"]

            n_channels = recon.shape[0]
            for c in range(n_channels):
                rec_c = recon[c].astype(np.float64, copy=False)
                tgt_c = target[c].astype(np.float64, copy=False)
                var_c = np.clip(
                    variance[c].astype(np.float64, copy=False),
                    VAR_CLIP[0],
                    VAR_CLIP[1],
                )
                sigma_c = np.sqrt(var_c)
                marker = str(marker_names[c])
                row = _patch_channel_row(
                    tgt_c.ravel(),
                    rec_c.ravel(),
                    var_c.ravel(),
                    sigma_c.ravel(),
                    marker=marker,
                    channel_id=int(channel_ids[c]),
                    meta=meta,
                    npz_path=npz_path,
                )
                rows.append(row)

                # Subsample pixels for the AUROC pool.
                abs_r_flat = np.abs(tgt_c.ravel() - rec_c.ravel())
                s_keep, r_keep = subsample_pc(
                    sigma_c.ravel(), abs_r_flat, rng, cap=per_pc_cap
                )
                pool_sigma.append(s_keep)
                pool_absr.append(r_keep)
                pool_marker.append(np.full(s_keep.shape, marker, dtype=object))
                pool_dataset.append(
                    np.full(s_keep.shape, str(meta.get("dataset_name", "")), dtype=object)
                )

    if pool_sigma:
        pool_df = pd.DataFrame(
            {
                "sigma": np.concatenate(pool_sigma).astype(np.float64),
                "abs_residual": np.concatenate(pool_absr).astype(np.float64),
                "marker": np.concatenate(pool_marker),
                "dataset": np.concatenate(pool_dataset),
            }
        )
        pool_df["marker"] = pool_df["marker"].astype("string")
        pool_df["dataset"] = pool_df["dataset"].astype("string")
    else:
        pool_df = pd.DataFrame(
            {"sigma": [], "abs_residual": [], "marker": [], "dataset": []}
        )

    return rows, pool_df


# ---------------------------------------------------------------------------
# AUSE — Area Under Sparsification Error.
# Sort pixels by predicted σ descending, sweep the kept fraction (1-f) from 1
# down to ~0; the error on the kept set is the sparsification curve. The
# oracle does the same, sorting by |r| instead of σ; SE = err_σ - err_oracle,
# AUSE = ∫ SE df. Lower is better. AUSE = 0 iff σ ranks pixels exactly like |r|.
# ---------------------------------------------------------------------------

def sparsification_curve(
    score: np.ndarray,
    abs_r: np.ndarray,
    n_points: int = 100,
    metric: str = "rmse",
) -> tuple[np.ndarray, np.ndarray]:
    """Error on the (1-f) lowest-`score` fraction of pixels, swept over f∈[0,1).

    Returns (fractions, err) where err[i] is the metric on the pixels kept
    after removing the highest-`score` fraction `fractions[i]`.
    """
    if metric not in ("rmse", "mae"):
        raise ValueError(f"metric must be 'rmse' or 'mae', got {metric!r}")
    score = np.asarray(score, dtype=np.float64).ravel()
    abs_r = np.asarray(abs_r, dtype=np.float64).ravel()
    if score.size != abs_r.size:
        raise ValueError("score and abs_r must have the same length")
    n = score.size
    if n == 0:
        return np.empty(0), np.empty(0)

    # Reorder abs_r so that index 0 has the *lowest* score (kept first).
    order_asc = np.argsort(score, kind="quicksort")
    r_asc = abs_r[order_asc]

    cum_sq = np.cumsum(r_asc * r_asc)
    cum_abs = np.cumsum(r_asc)
    counts = np.arange(1, n + 1, dtype=np.float64)

    fractions = np.linspace(0.0, 1.0, n_points + 1)[:-1]  # exclude empty kept set
    err = np.empty_like(fractions)
    for i, f in enumerate(fractions):
        k = max(1, int(round((1.0 - f) * n)))
        if metric == "rmse":
            err[i] = float(np.sqrt(cum_sq[k - 1] / counts[k - 1]))
        else:
            err[i] = float(cum_abs[k - 1] / counts[k - 1])
    return fractions, err


def ause(
    sigma: np.ndarray,
    abs_r: np.ndarray,
    metric: str = "rmse",
    n_points: int = 100,
    seed: int = 0,
) -> dict:
    """Compute AUSE plus the σ / oracle / random sparsification curves.

    Returns a dict with the four arrays (fractions, err_sigma, err_oracle,
    err_random), the sparsification-error curve, and three scalars
    (`ause`, `ause_random`, `ause_normalised`). All on the same pool.
    """
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    abs_r = np.asarray(abs_r, dtype=np.float64).ravel()

    fractions, e_sigma = sparsification_curve(sigma, abs_r, n_points, metric)
    _, e_oracle = sparsification_curve(abs_r, abs_r, n_points, metric)

    rng = np.random.default_rng(seed)
    rand_score = rng.permutation(abs_r)
    _, e_random = sparsification_curve(rand_score, abs_r, n_points, metric)

    se = e_sigma - e_oracle
    se_random = e_random - e_oracle
    ause_val = float(np.trapz(se, fractions))
    ause_rand = float(np.trapz(se_random, fractions))
    norm = float(ause_val / ause_rand) if ause_rand > 0 else float("nan")

    return {
        "fractions": fractions,
        "err_sigma": e_sigma,
        "err_oracle": e_oracle,
        "err_random": e_random,
        "sparsification_error": se,
        "ause": ause_val,
        "ause_random": ause_rand,
        "ause_normalised": norm,
        "metric": metric,
        "n_pixels": int(sigma.size),
    }
