"""Constants for calibration analysis."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


VAR_CLIP = (1e-6, 1e6)
LOGVAR_CLIP = (-15.0, 15.0)

ALPHA_GRID = np.round(np.arange(0.01, 1.00, 0.01), 3).astype(np.float64)
Z_ALPHA_GRID = norm.ppf((1.0 + ALPHA_GRID) / 2.0).astype(np.float64)

REPORT_ALPHAS = (0.5, 0.68, 0.9, 0.95, 0.99)

BINARY_MASK_QUANTILES = (0.90, 0.95, 0.99)

DEFAULT_AUROC_BUDGET = 2_000_000
DEFAULT_PER_PC_CAP = 1000
MIN_PIXELS_PER_MARKER_AUROC = 5_000

DEFAULT_NUM_WORKERS = 8
DEFAULT_SEED = 42


def alpha_col(alpha: float) -> str:
    return f"cov_count_{alpha:.3f}"


def quantile_cols(q: float) -> dict[str, str]:
    return {
        "tp": f"tp_q{q:.2f}",
        "fp": f"fp_q{q:.2f}",
        "fn": f"fn_q{q:.2f}",
        "tn": f"tn_q{q:.2f}",
        "sigma_thr": f"sigma_thr_q{q:.2f}",
        "r_thr": f"abs_r_thr_q{q:.2f}",
    }
