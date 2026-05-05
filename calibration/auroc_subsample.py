"""Stratified subsampling for pixel-level AUROC / AURC.

Per (patch, channel) we cap how many pixels we contribute, so a few patches
cannot dominate. The result is a pool of (sigma, |r|, marker, dataset) rows,
written to parquet for downstream metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_PER_PC_CAP


def subsample_pc(
    sigma: np.ndarray,
    abs_r: np.ndarray,
    rng: np.random.Generator,
    cap: int = DEFAULT_PER_PC_CAP,
) -> tuple[np.ndarray, np.ndarray]:
    """Return up to `cap` random pixels (without replacement) from a flat patch-channel."""
    n = sigma.size
    if n <= cap:
        return sigma, abs_r
    idx = rng.choice(n, size=cap, replace=False)
    return sigma[idx], abs_r[idx]


def concat_pools(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(
            columns=["sigma", "abs_residual", "marker", "dataset"]
        ).astype(
            {
                "sigma": np.float64,
                "abs_residual": np.float64,
                "marker": "string",
                "dataset": "string",
            }
        )
    return pd.concat(parts, ignore_index=True, copy=False)


def cap_pool(pool: pd.DataFrame, budget: int, rng: np.random.Generator) -> pd.DataFrame:
    """Trim pool to at most `budget` rows by uniform random subsample."""
    if len(pool) <= budget:
        return pool.reset_index(drop=True)
    idx = rng.choice(len(pool), size=budget, replace=False)
    return pool.iloc[idx].reset_index(drop=True)


def aurc(sigma: np.ndarray, sq_err: np.ndarray) -> float:
    """Area under the risk-coverage curve.

    Sort pixels by sigma ascending and compute cumulative MSE for the
    lowest-sigma fraction. Integrate over coverage in [0, 1].
    """
    if sigma.size == 0:
        return float("nan")
    order = np.argsort(sigma, kind="quicksort")
    sq_sorted = sq_err[order].astype(np.float64)
    cum = np.cumsum(sq_sorted)
    counts = np.arange(1, sq_sorted.size + 1, dtype=np.float64)
    risk = cum / counts
    coverage = counts / sq_sorted.size
    return float(np.trapz(risk, coverage))
