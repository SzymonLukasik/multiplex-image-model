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
