"""
Rotation Equivariance Evaluation Script (v3)

Implements the Tier 1 + key Tier 2 enhancements over evaluate_equivariance_v2_new.py
from EQUIVARIANCE_EVAL_SCRIPT_REVIEW.md:

  * Dense angle sweep with configurable grid (--angle-sweep)
  * Equivariance error vs. angle plot (the headline figure)
  * Bootstrap confidence intervals on all aggregated metrics
  * Interpolation-error baseline (rotate-then-inverse without model)
    decoupled from model-attributable equivariance error
  * GPU-side metric computation for speed
  * Per-channel Pearson distribution histogram
  * Markdown summary report alongside CSV/JSON
  * Environment + checkpoint hashing for reproducibility
  * Streaming aggregation to avoid per-sample OOM
  * Dynamic reconstruction crop (no hardcoded [3:-4])
  * Configurable inference precision

The script saves to <output_dir>/<checkpoint_name>/ with:
  metrics.csv             # per-sample, per-angle metrics
  aggregated.json         # mean / std / CI per angle
  report.md               # human-readable summary
  plots/
    angle_curve_*.png     # equivariance error vs. angle (per metric)
    channel_pearson_hist_*.png
    interpolation_baseline.png
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm

from load_model_for_evaluation import load_model_and_data


# =============================================================================
# Rotation / inverse-rotation utilities
# =============================================================================

def rotate_tensor(x: torch.Tensor, theta_deg: float) -> torch.Tensor:
    """Rotate (B, C, H, W) around the center. Exact for 90° multiples; bilinear otherwise."""
    if abs(theta_deg) % 90 == 0:
        k = int(round(theta_deg / 90)) % 4
        return torch.rot90(x, k=k, dims=(-2, -1))
    return TF.rotate(
        x, theta_deg,
        interpolation=TF.InterpolationMode.BILINEAR,
        expand=False,
        center=((x.shape[-1] - 1) / 2, (x.shape[-2] - 1) / 2),
    )


def apply_transformation(img: torch.Tensor, rotation: float, flip: bool) -> torch.Tensor:
    if rotation != 0:
        img = rotate_tensor(img, rotation)
    if flip:
        img = torch.flip(img, dims=[-1])
    return img


def inverse_transformation(x: torch.Tensor, rotation: float, flip: bool) -> torch.Tensor:
    if flip:
        x = torch.flip(x, dims=[-1])
    if rotation != 0:
        x = rotate_tensor(x, -rotation)
    return x


def central_circle_mask(H: int, W: int, device: torch.device, radius_fraction: float = 0.7) -> torch.Tensor:
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    Y, X = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij',
    )
    r = ((X - cx) ** 2 + (Y - cy) ** 2).sqrt()
    max_r = min(H, W) / 2.0 * radius_fraction
    return (r <= max_r).float()


# =============================================================================
# GPU-side metrics
# =============================================================================

def _pearson_flat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pearson correlation per batch element. a, b: (B, ...) flattened."""
    a_flat = a.flatten(1)
    b_flat = b.flatten(1)
    ac = a_flat - a_flat.mean(dim=1, keepdim=True)
    bc = b_flat - b_flat.mean(dim=1, keepdim=True)
    return F.cosine_similarity(ac, bc, dim=1)


def _per_channel_pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-channel Pearson, shape (B, C)."""
    B, C = a.shape[:2]
    a_flat = a.reshape(B, C, -1)
    b_flat = b.reshape(B, C, -1)
    ac = a_flat - a_flat.mean(dim=2, keepdim=True)
    bc = b_flat - b_flat.mean(dim=2, keepdim=True)
    num = (ac * bc).sum(dim=2)
    den = ac.norm(dim=2) * bc.norm(dim=2) + 1e-8
    return num / den


@dataclass
class EquivMetrics:
    """Per-sample metrics for one (transformation, batch)."""
    equiv_mse: np.ndarray              # (B,)
    relative_mse: np.ndarray
    pearson: np.ndarray
    central_relative_mse: np.ndarray
    central_pearson: np.ndarray
    per_channel_pearson: np.ndarray    # (B, C)


def compute_latent_equiv_metrics(
    latent_inv: torch.Tensor, reference: torch.Tensor, radius_fraction: float = 0.7,
) -> EquivMetrics:
    """All metrics on GPU; .cpu().numpy() only at the boundary."""
    B, C, H, W = latent_inv.shape
    device = latent_inv.device

    diff_sq = (latent_inv - reference) ** 2
    equiv_mse = diff_sq.mean(dim=[1, 2, 3])
    ref_energy = (reference ** 2).mean(dim=[1, 2, 3]) + 1e-8
    relative_mse = equiv_mse / ref_energy
    pearson = _pearson_flat(latent_inv, reference)

    c_mask = central_circle_mask(H, W, device, radius_fraction).unsqueeze(0).unsqueeze(0)
    n_pix = c_mask.sum().clamp(min=1)
    cent_diff_sq = diff_sq * c_mask
    cent_mse = cent_diff_sq.sum(dim=[2, 3]) / n_pix
    cent_mse = cent_mse.mean(dim=1)
    cent_ref_energy = ((reference ** 2) * c_mask).sum(dim=[2, 3]) / n_pix
    cent_ref_energy = cent_ref_energy.mean(dim=1) + 1e-8
    cent_rel_mse = cent_mse / cent_ref_energy
    cent_pearson = _pearson_flat(latent_inv * c_mask, reference * c_mask)

    pc_pearson = _per_channel_pearson(latent_inv, reference)

    return EquivMetrics(
        equiv_mse=equiv_mse.detach().cpu().numpy(),
        relative_mse=relative_mse.detach().cpu().numpy(),
        pearson=pearson.detach().cpu().numpy(),
        central_relative_mse=cent_rel_mse.detach().cpu().numpy(),
        central_pearson=cent_pearson.detach().cpu().numpy(),
        per_channel_pearson=pc_pearson.detach().cpu().numpy(),
    )


def compute_recon_consistency_mse(
    recon: torch.Tensor, recon_ref_rotated: torch.Tensor, radius_fraction: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray]:
    """Recon consistency: D(E(T(x))) vs T(D(E(x))). Returns (full_mse, central_mse)."""
    B, C, H, W = recon.shape
    diff_sq = (recon - recon_ref_rotated) ** 2
    full_mse = diff_sq.mean(dim=[1, 2, 3])
    c_mask = central_circle_mask(H, W, recon.device, radius_fraction).unsqueeze(0).unsqueeze(0)
    n_pix = c_mask.sum().clamp(min=1)
    cent_mse = (diff_sq * c_mask).sum(dim=[2, 3]) / n_pix
    cent_mse = cent_mse.mean(dim=1)
    return full_mse.detach().cpu().numpy(), cent_mse.detach().cpu().numpy()


def compute_interpolation_baseline(
    img: torch.Tensor, rotation: float, flip: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate then inverse-rotate the *input* (no model in between).
    Quantifies how much MSE / Pearson degradation is caused purely by
    bilinear interpolation artifacts. Subtract from model error to get
    the model-attributable equivariance error.
    """
    if rotation % 90 == 0 and not flip:
        # Exact: torch.rot90 is invertible without loss.
        zeros = torch.zeros(img.shape[0], device=img.device)
        ones = torch.ones(img.shape[0], device=img.device)
        return zeros.cpu().numpy(), ones.cpu().numpy()
    transformed = apply_transformation(img, rotation, flip)
    recovered = inverse_transformation(transformed, rotation, flip)
    diff_sq = (img - recovered) ** 2
    mse = diff_sq.mean(dim=[1, 2, 3])
    pear = _pearson_flat(img, recovered)
    return mse.detach().cpu().numpy(), pear.detach().cpu().numpy()


# =============================================================================
# Streaming aggregator
# =============================================================================

class StreamingStats:
    """Running per-sample value accumulator. Keeps the full per-sample array
    in memory (needed for bootstrap CI and Wilcoxon); strips heavy arrays
    (reconstructions, latents) before storing."""

    def __init__(self):
        self.values: Dict[str, List[np.ndarray]] = {}

    def add(self, key: str, values: np.ndarray):
        self.values.setdefault(key, []).append(np.asarray(values).reshape(-1))

    def flatten(self, key: str) -> Optional[np.ndarray]:
        chunks = self.values.get(key)
        if not chunks:
            return None
        return np.concatenate(chunks, axis=0)


def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95,
                 seed: int = 0) -> Tuple[float, float]:
    if len(values) == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    boot_means = np.nanmean(values[idx], axis=1)
    lo = float(np.percentile(boot_means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot_means, (1 + ci) / 2 * 100))
    return lo, hi


def aggregate_with_ci(
    stats: StreamingStats, n_bootstrap: int = 1000, seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key in sorted(stats.values.keys()):
        vals = stats.flatten(key)
        if vals is None or len(vals) == 0:
            continue
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            continue
        mean = float(np.nanmean(vals))
        std = float(np.nanstd(vals))
        median = float(np.nanmedian(vals))
        lo, hi = bootstrap_ci(finite, n_bootstrap=n_bootstrap, seed=seed)
        out[key] = {
            'mean': mean, 'std': std, 'median': median,
            'ci95_lo': lo, 'ci95_hi': hi,
            'n_samples': int(len(vals)),
            'n_finite': int(len(finite)),
        }
    return out


# =============================================================================
# Per-batch evaluation
# =============================================================================

@dataclass
class TransformSpec:
    rotation: float
    flip: bool

    @property
    def key(self) -> str:
        sign = 'f' if self.flip else 'nf'
        return f"r{self.rotation:g}_{sign}"


def _dynamic_crop(big: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Center-crop big to (target_h, target_w). Avoids the hardcoded [3:-4]."""
    h, w = big.shape[-2:]
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return big[..., top:top + target_h, left:left + target_w]


def _to_plain(t):
    """Convert escnn.GeometricTensor → torch.Tensor; pass through tensors."""
    return t.tensor if hasattr(t, 'tensor') else t


def _forward(model, img, channel_ids, autocast_ctx):
    """Single forward; returns latent (last layer) and full reconstruction mean."""
    with torch.no_grad(), autocast_ctx:
        output = model(img, channel_ids, channel_ids, True)
        features = output["features"]
        recon = output["output"]
        latent = _to_plain(features[-1]).float()
        # recon is (B, C, H, W, 2): (mean_logit, logsigma). Pull mean only.
        mi_full = torch.sigmoid(recon[..., 0]).float()
    return latent, mi_full


def evaluate_batch(
    model, img: torch.Tensor, channel_ids: torch.Tensor, transforms: Sequence[TransformSpec],
    autocast_ctx, radius_fraction: float = 0.7,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Evaluate one batch. Returns:
      results[transform_key] = {
        'equiv_mse': (B,), 'relative_mse': (B,), 'pearson': (B,),
        'central_relative_mse': (B,), 'central_pearson': (B,),
        'recon_consistency_mse': (B,), 'recon_consistency_central_mse': (B,),
        'interp_baseline_mse': (B,), 'interp_baseline_pearson': (B,),
        'model_attrib_mse': (B,),   # equiv_mse - interp baseline (clipped >= 0)
        'per_channel_pearson': (B, C)
      }
    """
    device = img.device
    ref_latent, ref_recon = _forward(model, img, channel_ids, autocast_ctx)

    results: Dict[str, Dict[str, np.ndarray]] = {}

    for t in transforms:
        if t.rotation == 0 and not t.flip:
            continue  # identity, skip

        img_t = apply_transformation(img, t.rotation, t.flip)
        latent_t, recon_t = _forward(model, img_t, channel_ids, autocast_ctx)

        # Inverse-transform the latent to align with reference
        latent_inv = inverse_transformation(latent_t, t.rotation, t.flip)
        metrics = compute_latent_equiv_metrics(latent_inv, ref_latent, radius_fraction)

        # Recon consistency: D(E(T(x))) vs T(D(E(x)))
        ref_recon_rotated = apply_transformation(ref_recon, t.rotation, t.flip)
        recon_mse, recon_cent_mse = compute_recon_consistency_mse(
            recon_t, ref_recon_rotated, radius_fraction
        )

        # Interpolation-only baseline (input → rotated → inverse-rotated)
        interp_mse, interp_pearson = compute_interpolation_baseline(img, t.rotation, t.flip)

        model_attrib = np.maximum(metrics.equiv_mse - interp_mse, 0.0)

        results[t.key] = {
            'equiv_mse': metrics.equiv_mse,
            'relative_mse': metrics.relative_mse,
            'pearson': metrics.pearson,
            'central_relative_mse': metrics.central_relative_mse,
            'central_pearson': metrics.central_pearson,
            'recon_consistency_mse': recon_mse,
            'recon_consistency_central_mse': recon_cent_mse,
            'interp_baseline_mse': interp_mse,
            'interp_baseline_pearson': interp_pearson,
            'model_attrib_mse': model_attrib,
            'per_channel_pearson': metrics.per_channel_pearson,
        }
    return results


# =============================================================================
# Plotting
# =============================================================================

PLOT_METRICS = {
    'central_relative_mse': 'Central Relative Equivariance MSE',
    'pearson': 'Pearson correlation (latent)',
    'central_pearson': 'Central Pearson correlation',
    'recon_consistency_mse': 'Reconstruction consistency MSE',
    'recon_consistency_central_mse': 'Reconstruction consistency MSE (central)',
    'model_attrib_mse': 'Model-attributable equivariance MSE',
    'interp_baseline_mse': 'Interpolation-only baseline MSE',
}


def plot_angle_curve(
    aggregated: Dict[str, Dict[str, float]], metric: str, save_path: Path,
    title_suffix: str = '',
):
    """Plot one metric vs. rotation angle with CI band."""
    rows: List[Tuple[float, bool, float, float, float]] = []
    for key, stats in aggregated.items():
        if not key.startswith('r'):
            continue
        # parse "r<angle>_<f|nf>"
        try:
            angle_str, sign = key[1:].rsplit('_', 1)
            angle = float(angle_str)
            flip = (sign == 'f')
        except ValueError:
            continue
        rows.append((angle, flip, stats[f'{metric}_mean'], stats[f'{metric}_ci95_lo'], stats[f'{metric}_ci95_hi']))

    if not rows:
        return
    rows.sort(key=lambda r: (r[1], r[0]))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for flip_val, marker, label in [(False, 'o', 'no flip'), (True, 's', 'flip')]:
        sel = [r for r in rows if r[1] == flip_val]
        if not sel:
            continue
        a = np.array([r[0] for r in sel])
        m = np.array([r[2] for r in sel])
        lo = np.array([r[3] for r in sel])
        hi = np.array([r[4] for r in sel])
        ax.plot(a, m, marker=marker, label=label)
        ax.fill_between(a, lo, hi, alpha=0.2)

    for ax_x in (0, 90, 180, 270, 360):
        ax.axvline(x=ax_x, ls='--', color='grey', alpha=0.3)

    ax.set_xlabel('Rotation angle (degrees)')
    ax.set_ylabel(PLOT_METRICS.get(metric, metric))
    ax.set_title(f'{PLOT_METRICS.get(metric, metric)}{title_suffix}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_channel_pearson_hist(
    stats: StreamingStats, transform_key: str, save_path: Path,
):
    """Per-channel Pearson histogram for one transformation."""
    flat = stats.flatten(f'{transform_key}/per_channel_pearson_flat')
    if flat is None or len(flat) == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(flat, bins=50, edgecolor='black', linewidth=0.5)
    ax.axvline(np.nanmedian(flat), color='red', ls='--', label=f'median = {np.nanmedian(flat):.3f}')
    ax.set_xlabel('Per-channel Pearson')
    ax.set_ylabel('Count')
    ax.set_title(f'Per-channel Pearson @ {transform_key}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_interp_baseline(aggregated: Dict[str, Dict[str, float]], save_path: Path):
    """Compare equiv_mse vs interpolation-only baseline at each angle."""
    rows: List[Tuple[float, bool, float, float]] = []
    for key, stats in aggregated.items():
        if not key.startswith('r'):
            continue
        try:
            angle_str, sign = key[1:].rsplit('_', 1)
            angle = float(angle_str)
            flip = (sign == 'f')
        except ValueError:
            continue
        e = stats.get('equiv_mse_mean', float('nan'))
        b = stats.get('interp_baseline_mse_mean', float('nan'))
        rows.append((angle, flip, e, b))
    if not rows:
        return
    rows.sort(key=lambda r: (r[1], r[0]))
    no_flip = [r for r in rows if not r[1]]
    if not no_flip:
        return
    a = np.array([r[0] for r in no_flip])
    e = np.array([r[2] for r in no_flip])
    b = np.array([r[3] for r in no_flip])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(a, e, marker='o', label='Total equiv. MSE (latent)')
    ax.plot(a, b, marker='s', label='Interpolation-only baseline (input)')
    ax.plot(a, np.maximum(e - b, 0), marker='^', label='Model-attributable (e - b)')
    ax.set_xlabel('Rotation angle (degrees)')
    ax.set_ylabel('MSE')
    ax.set_title('Equivariance error decomposition (no flip)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# =============================================================================
# Report writer
# =============================================================================

def _hash_file(path: str) -> str:
    if not path or not os.path.isfile(path):
        return ''
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ''


def _env_info(config_path: str, checkpoint_path: str) -> Dict[str, str]:
    info = {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'torch': torch.__version__,
        'cuda_available': str(torch.cuda.is_available()),
        'cuda_version': torch.version.cuda or '',
        'git_sha': _git_sha(),
        'config_path': config_path,
        'config_sha256_16': _hash_file(config_path),
        'checkpoint_path': checkpoint_path,
        'checkpoint_sha256_16': _hash_file(checkpoint_path),
        'eval_date_utc': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
    }
    try:
        import escnn  # type: ignore
        info['escnn'] = escnn.__version__ if hasattr(escnn, '__version__') else 'unknown'
    except ImportError:
        info['escnn'] = 'not installed'
    return info


def write_report(
    output_dir: Path, aggregated: Dict[str, Dict[str, float]], env: Dict[str, str],
    args: argparse.Namespace, n_samples: int,
):
    """Write a Markdown summary report. Designed to be paste-able into Slack."""
    lines: List[str] = []
    lines.append("# Equivariance Evaluation Report (v3)\n")
    lines.append("## Environment\n")
    for k, v in env.items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("")
    lines.append("## Arguments\n")
    for k, v in vars(args).items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("")
    lines.append(f"## Samples evaluated\n\n{n_samples}\n")

    # Headline summary: model_attrib_mse and central_pearson at key angles
    key_angles = [22.5, 30, 45, 60, 90, 135, 180]
    summary_rows: List[Tuple[str, str, str, str, str]] = []
    summary_rows.append(("Angle", "Model-attrib. MSE [95% CI]", "Central Pearson [95% CI]", "Recon. consistency MSE", "Interp. baseline MSE"))
    for angle in key_angles:
        key = f"r{angle:g}_nf"
        st = aggregated.get(key)
        if not st:
            continue
        def fmt(m, ci_lo, ci_hi):
            val = st.get(f'{m}_mean', float('nan'))
            lo = st.get(f'{m}_ci95_lo', float('nan'))
            hi = st.get(f'{m}_ci95_hi', float('nan'))
            return f"{val:.4g} [{lo:.4g}, {hi:.4g}]"
        summary_rows.append((
            f"{angle:g}°",
            fmt('model_attrib_mse', 'ci95_lo', 'ci95_hi'),
            fmt('central_pearson', 'ci95_lo', 'ci95_hi'),
            fmt('recon_consistency_mse', 'ci95_lo', 'ci95_hi'),
            fmt('interp_baseline_mse', 'ci95_lo', 'ci95_hi'),
        ))
    lines.append("## Summary (no-flip rotations)\n")
    header, *body = summary_rows
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Files\n")
    lines.append("- `metrics.csv` — per-sample, per-angle metrics for offline analysis")
    lines.append("- `aggregated.json` — mean / std / CI per angle and metric")
    lines.append("- `plots/angle_curve_*.png` — equivariance error vs. rotation angle")
    lines.append("- `plots/channel_pearson_hist_*.png` — per-channel Pearson distribution")
    lines.append("- `plots/interpolation_baseline.png` — decomposes equivariance error into interpolation vs. model-attributable")
    lines.append("")

    (output_dir / 'report.md').write_text('\n'.join(lines))


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Equivariance evaluation v3 (rigor pass)')
    p.add_argument('--config', type=str, required=True, help='Training config YAML')
    p.add_argument('--checkpoint', type=str, default=None, help='Override config.from_checkpoint')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output-dir', type=str, default='equivariance_results_v3')
    p.add_argument('--num-batches', type=int, default=None,
                   help='Limit number of test batches (None = all)')
    p.add_argument('--fraction', type=float, default=1.0,
                   help='Fraction of batches to subsample (random seed via --seed)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--precision', choices=['fp32', 'bf16', 'fp16'], default='fp32',
                   help='Inference precision; fp32 is safest for equivariance.')

    # Angle sweep
    p.add_argument('--angle-start', type=float, default=0.0)
    p.add_argument('--angle-stop', type=float, default=360.0)
    p.add_argument('--angle-step', type=float, default=15.0)
    p.add_argument('--extra-angles', type=float, nargs='*', default=[22.5, 45, 67.5, 112.5, 135, 157.5],
                   help='Extra angles to include alongside the uniform sweep (off-grid).')
    p.add_argument('--include-flips', action='store_true',
                   help='Also evaluate each angle with horizontal flip.')

    # Bootstrap
    p.add_argument('--bootstrap-iters', type=int, default=1000)

    # Plotting / output
    p.add_argument('--central-radius', type=float, default=0.7,
                   help='Central-circle mask radius fraction for boundary-free metrics.')
    p.add_argument('--channel-hist-angles', type=float, nargs='*', default=[45.0],
                   help='Angles at which to plot per-channel Pearson histograms.')
    return p.parse_args()


def build_transforms(args) -> List[TransformSpec]:
    angles = list(np.arange(args.angle_start, args.angle_stop, args.angle_step))
    angles += args.extra_angles
    angles = sorted({round(float(a), 4) for a in angles})
    transforms: List[TransformSpec] = []
    for a in angles:
        transforms.append(TransformSpec(rotation=a, flip=False))
        if args.include_flips:
            transforms.append(TransformSpec(rotation=a, flip=True))
    return transforms


def build_autocast(precision: str, device: str):
    if precision == 'fp32' or device == 'cpu':
        return nullcontext()
    dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
    return torch.amp.autocast(device_type='cuda', dtype=dtype)


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    transforms = build_transforms(args)
    print(f"[v3] {len(transforms)} transformations: angle grid step={args.angle_step}°, "
          f"flips={'yes' if args.include_flips else 'no'}")

    model, _, test_dataloader, _, _, config = load_model_and_data(
        config_path=args.config, checkpoint_path=args.checkpoint, device=args.device,
    )
    model.eval()
    autocast_ctx = build_autocast(args.precision, args.device)

    # Output directory
    ckpt_path = args.checkpoint or config.get('from_checkpoint', '') or ''
    ckpt_name = (
        os.path.basename(ckpt_path).replace('.pth', '') if ckpt_path
        else f"no_checkpoint_{int(time.time())}"
    )
    output_dir = Path(args.output_dir) / ckpt_name
    (output_dir / 'plots').mkdir(parents=True, exist_ok=True)
    print(f"[v3] output_dir = {output_dir}")

    env = _env_info(args.config, ckpt_path)
    print(f"[v3] env: torch={env['torch']} cuda={env['cuda_version']} git={env['git_sha']}")

    # Batch selection
    total_batches = len(test_dataloader)
    if args.num_batches:
        total_batches = min(args.num_batches, total_batches)
    if args.fraction < 1.0:
        n_use = max(1, int(total_batches * args.fraction))
        sel = set(np.random.RandomState(args.seed).choice(total_batches, n_use, replace=False))
    else:
        n_use = total_batches
        sel = None

    # Streaming aggregator
    stats = StreamingStats()
    csv_rows: List[Dict[str, str]] = []
    n_samples_total = 0
    metric_names = (
        'equiv_mse', 'relative_mse', 'pearson',
        'central_relative_mse', 'central_pearson',
        'recon_consistency_mse', 'recon_consistency_central_mse',
        'interp_baseline_mse', 'interp_baseline_pearson',
        'model_attrib_mse',
    )

    for i, (img, channel_ids, panel_idx, img_path) in enumerate(tqdm(
        test_dataloader, total=n_use, desc='Eval'
    )):
        if i >= total_batches:
            break
        if sel is not None and i not in sel:
            continue

        img = img.to(args.device).float()
        channel_ids = channel_ids.to(args.device).long()
        n_samples_total += img.shape[0]

        results = evaluate_batch(model, img, channel_ids, transforms, autocast_ctx,
                                 radius_fraction=args.central_radius)

        # Accumulate per-sample metric arrays and CSV rows
        batch_size = img.shape[0]
        paths = img_path if isinstance(img_path, (list, tuple)) else [str(img_path)] * batch_size
        for key, m in results.items():
            for mn in metric_names:
                if mn in m:
                    stats.add(f'{key}/{mn}', m[mn])
            # per-channel pearson — flatten all (sample, channel) values for histogram
            if 'per_channel_pearson' in m:
                stats.add(f'{key}/per_channel_pearson_flat', m['per_channel_pearson'].reshape(-1))
            # CSV rows
            for s in range(batch_size):
                row = {
                    'sample_id': paths[s] if s < len(paths) else f'batch{i}_s{s}',
                    'batch_idx': i,
                    'sample_in_batch': s,
                    'transformation': key,
                }
                for mn in metric_names:
                    if mn in m and s < len(m[mn]):
                        row[mn] = float(m[mn][s])
                csv_rows.append(row)

    # Aggregate
    print(f"[v3] Aggregating with bootstrap CI ({args.bootstrap_iters} iters)...")
    aggregated_raw = aggregate_with_ci(stats, n_bootstrap=args.bootstrap_iters, seed=args.seed)

    # Reshape: keyed by transform_key with metric subkeys
    aggregated: Dict[str, Dict[str, float]] = {}
    for full_key, st in aggregated_raw.items():
        if '/' not in full_key:
            continue
        transform_key, metric = full_key.split('/', 1)
        if metric == 'per_channel_pearson_flat':
            continue  # used only for histogram
        if transform_key not in aggregated:
            aggregated[transform_key] = {}
        for s_name, s_val in st.items():
            aggregated[transform_key][f'{metric}_{s_name}'] = s_val

    # Persist
    with open(output_dir / 'aggregated.json', 'w') as f:
        json.dump({'env': env, 'args': vars(args), 'aggregated': aggregated}, f, indent=2, default=str)

    if csv_rows:
        fieldnames = ['sample_id', 'batch_idx', 'sample_in_batch', 'transformation'] + list(metric_names)
        with open(output_dir / 'metrics.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)

    # Plots
    plots_dir = output_dir / 'plots'
    for metric in PLOT_METRICS:
        if any(f'{metric}_mean' in v for v in aggregated.values()):
            plot_angle_curve(aggregated, metric, plots_dir / f'angle_curve_{metric}.png')
    plot_interp_baseline(aggregated, plots_dir / 'interpolation_baseline.png')
    for a in args.channel_hist_angles:
        key = f"r{a:g}_nf"
        if key in {k for k in stats.values if k.startswith('r')}:
            plot_channel_pearson_hist(stats, key, plots_dir / f'channel_pearson_hist_{a:g}.png')

    # Report
    write_report(output_dir, aggregated, env, args, n_samples_total)
    print(f"[v3] Done. Results in {output_dir}")


if __name__ == '__main__':
    main()
