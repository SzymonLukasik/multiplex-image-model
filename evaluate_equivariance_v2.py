"""
Rotation Equivariance Evaluation Script (v2)

Evaluates the model's rotation equivariance by:
1. Testing various rotations (90°, 180°, 270°, 30°, 45°, 135°) and flips
2. Computing equivariance errors between latent representations
3. Calculating metrics including Relative MSE, Pearson correlation,
   central-region metrics, and reconstruction consistency
4. Separating results into exact (90°-multiple) and approximate rotation groups

Changes from v1:
- Added Pearson correlation (centered cosine similarity) as primary metric
- Added channelwise Pearson for diagnostics
- Added central circular mask metrics (boundary-free evaluation)
- Added reconstruction consistency test: T(D(E(x))) vs D(E(T(x)))
- Removed input-level circular masking (no more masked reference pass)
- Removed r0_nf_mask; always compare against unmasked original
- Added EXACT_MEAN / APPROX_MEAN summary groups
"""

import argparse
import csv
import json
import math
import os
from math import ceil
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm
from contextlib import nullcontext

from load_model_for_evaluation import load_model_and_data

# PyTorch version compatibility for autocast
try:
    from torch.cuda.amp import autocast as cuda_autocast
    HAS_AUTOCAST = True
except ImportError:
    HAS_AUTOCAST = False


def rotate_tensor(x: torch.Tensor, theta_deg: float, mode: str = 'bilinear') -> torch.Tensor:
    """
    Rotate a tensor by theta_deg degrees around the center.

    Args:
        x: Input tensor [B, C, H, W]
        theta_deg: Rotation angle in degrees
        mode: Interpolation mode ('bilinear' or 'nearest')

    Returns:
        Rotated tensor [B, C, H, W]
    """
    if abs(theta_deg) % 90 == 0:
        k = int(round(theta_deg / 90)) % 4
        return torch.rot90(x, k=k, dims=(-2, -1))

    interp_mode = (
        TF.InterpolationMode.BILINEAR if mode == 'bilinear'
        else TF.InterpolationMode.NEAREST
    )
    return TF.rotate(
        x, theta_deg,
        interpolation=interp_mode,
        expand=False,
        center=((x.shape[-1] - 1) / 2, (x.shape[-2] - 1) / 2)
    )


def compute_cosine_similarity(latent1: torch.Tensor, latent2: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between two latent representations.

    Args:
        latent1: First latent [B, C, H, W]
        latent2: Second latent [B, C, H, W]

    Returns:
        Cosine similarity per sample [B]
    """
    flat1 = latent1.flatten(1)
    flat2 = latent2.flatten(1)
    return F.cosine_similarity(flat1, flat2, dim=1)


def compute_pearson_correlation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute Pearson correlation (centered cosine similarity) between two tensors.
    Unlike cosine similarity, this is invariant to activation magnitude.
    Reference: Bruintjes et al. CVPR-W 2023.

    Args:
        a: First tensor [B, C, H, W]
        b: Second tensor [B, C, H, W]

    Returns:
        Pearson correlation per sample [B]
    """
    a_flat = a.flatten(1)
    b_flat = b.flatten(1)
    a_centered = a_flat - a_flat.mean(dim=1, keepdim=True)
    b_centered = b_flat - b_flat.mean(dim=1, keepdim=True)
    return F.cosine_similarity(a_centered, b_centered, dim=1)


def compute_channelwise_pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute per-channel Pearson correlation, averaged over channels.
    Useful for diagnosing which channels break equivariance.

    Args:
        a: First tensor [B, C, H, W]
        b: Second tensor [B, C, H, W]

    Returns:
        Mean channelwise Pearson per sample [B]
    """
    B, C, H, W = a.shape
    a_flat = a.reshape(B, C, -1)
    b_flat = b.reshape(B, C, -1)
    a_centered = a_flat - a_flat.mean(dim=2, keepdim=True)
    b_centered = b_flat - b_flat.mean(dim=2, keepdim=True)
    num = (a_centered * b_centered).sum(dim=2)
    den = a_centered.norm(dim=2) * b_centered.norm(dim=2) + 1e-8
    per_channel = num / den
    return per_channel.mean(dim=1)


def central_circle_mask(H: int, W: int, device: torch.device,
                        radius_fraction: float = 0.7) -> torch.Tensor:
    """
    Create a circular mask covering the central region of a feature map.
    Used to exclude boundary-contaminated features from equivariance measurement.
    radius_fraction=0.7 covers ~49% of the spatial area.

    Args:
        H: Height of feature map
        W: Width of feature map
        device: Device
        radius_fraction: Fraction of half-side-length to use as radius

    Returns:
        Binary mask [H, W]
    """
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    Y, X = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    r = ((X - cx) ** 2 + (Y - cy) ** 2).sqrt()
    max_r = min(H, W) / 2.0 * radius_fraction
    return (r <= max_r).float()


def compute_equivariance_metrics(
    latent_inv: torch.Tensor,
    reference: torch.Tensor,
    central_radius_fraction: float = 0.7
) -> Dict[str, np.ndarray]:
    """
    Compute full suite of equivariance metrics between inverse-transformed
    latent and reference latent, for both full spatial extent and central region.

    Args:
        latent_inv: Inverse-transformed latent [B, C, H, W]
        reference: Reference latent [B, C, H, W]
        central_radius_fraction: Radius fraction for central mask

    Returns:
        Dict of metric_name -> numpy array [B]
    """
    metrics = {}
    B, C, H, W = latent_inv.shape

    # --- Full-latent metrics ---
    diff_sq = (latent_inv - reference) ** 2

    # Raw MSE
    equiv_mse = diff_sq.mean(dim=[1, 2, 3])
    metrics['equiv_mse'] = equiv_mse.numpy()

    # Relative MSE (normalized by reference energy)
    ref_energy = (reference ** 2).mean(dim=[1, 2, 3]) + 1e-8
    relative_mse = equiv_mse / ref_energy
    metrics['relative_mse'] = relative_mse.numpy()

    # L1
    equiv_l1 = (latent_inv - reference).abs().mean(dim=[1, 2, 3])
    metrics['equiv_l1'] = equiv_l1.numpy()

    # Cosine similarity (kept for backward compatibility)
    metrics['cosine_sim'] = compute_cosine_similarity(latent_inv, reference).numpy()

    # Pearson correlation (primary similarity metric)
    metrics['pearson'] = compute_pearson_correlation(latent_inv, reference).numpy()

    # Channelwise Pearson (diagnostic)
    metrics['channelwise_pearson'] = compute_channelwise_pearson(latent_inv, reference).numpy()

    # --- Central-region metrics ---
    c_mask = central_circle_mask(H, W, latent_inv.device, central_radius_fraction)
    mask_expanded = c_mask.unsqueeze(0).unsqueeze(0)
    n_pixels = c_mask.sum().clamp(min=1)

    # Central MSE
    central_diff_sq = diff_sq * mask_expanded
    central_mse = central_diff_sq.sum(dim=[2, 3]) / n_pixels
    central_mse = central_mse.mean(dim=1)
    metrics['central_mse'] = central_mse.numpy()

    # Central relative MSE
    central_ref_energy = ((reference ** 2) * mask_expanded).sum(dim=[2, 3]) / n_pixels
    central_ref_energy = central_ref_energy.mean(dim=1) + 1e-8
    metrics['central_relative_mse'] = (central_mse / central_ref_energy).numpy()

    # Central Pearson
    latent_masked = latent_inv * mask_expanded
    ref_masked = reference * mask_expanded
    metrics['central_pearson'] = compute_pearson_correlation(latent_masked, ref_masked).numpy()

    return metrics


def input_circle_mask(H: int, W: int, device: torch.device) -> torch.Tensor:
    """
    Create a binary mask for the largest inscribed circle in an H x W image.
    Used to mask inputs before continuous rotations so that no zero-filled
    corners appear after rotation (the circle is rotationally invariant).

    Args:
        H: Image height
        W: Image width
        device: Device

    Returns:
        Binary mask [1, 1, H, W] ready for broadcasting with [B, C, H, W] images
    """
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    Y, X = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    r = ((X - cx) ** 2 + (Y - cy) ** 2).sqrt()
    max_r = min(H, W) / 2.0
    return (r <= max_r).float().unsqueeze(0).unsqueeze(0)


def apply_transformation(
    img: torch.Tensor,
    rotation: float,
    flip: bool,
    device: torch.device
) -> torch.Tensor:
    """
    Apply rotation and optional flip to an image.
    Non-90° rotations use bilinear interpolation (matching vanilla training augmentation).
    No input masking — boundary artifacts are part of the practical evaluation.

    Args:
        img: Input image [B, C, H, W]
        rotation: Rotation angle in degrees
        flip: Whether to apply horizontal flip
        device: Device for computation

    Returns:
        Transformed image [B, C, H, W]
    """
    if rotation != 0:
        if rotation % 90 == 0:
            img = torch.rot90(img, k=int(rotation // 90) % 4, dims=[-2, -1])
        else:
            img = rotate_tensor(img, rotation, mode='bilinear')

    if flip:
        img = torch.flip(img, dims=[3])

    return img


def inverse_transformation(
    latent: torch.Tensor,
    rotation: float,
    flip: bool
) -> torch.Tensor:
    """
    Apply inverse transformation to latent representation.

    Args:
        latent: Latent tensor [B, C, H, W]
        rotation: Original rotation angle in degrees
        flip: Whether flip was applied

    Returns:
        Inversely transformed latent [B, C, H, W]
    """
    if flip:
        latent = torch.flip(latent, dims=[3])

    if rotation > 0 and rotation % 90 == 0:
        latent = torch.rot90(latent, k=int(-rotation // 90), dims=[-2, -1])
    elif rotation > 0:
        latent = rotate_tensor(latent, -rotation, mode='bilinear')

    return latent


def to_tensor(x):
    """Convert GeometricTensor or regular tensor to regular PyTorch tensor."""
    if hasattr(x, 'tensor'):
        return x.tensor
    return x


def _run_model_pass(model, img, channel_ids_batch, autocast_context):
    """Run a single forward pass and return latent features + reconstruction."""
    with torch.no_grad():
        with autocast_context:
            output = model(img, channel_ids_batch, channel_ids_batch, True)
            features = output["features"]
            recon = output["output"]
            recon = recon[:, :, 3:-4, 3:-4]
            mi, logsigma = recon.unbind(dim=-1)
            mi = torch.sigmoid(mi)
    return features, mi, logsigma


def _compute_transform_metrics(
    features, mi, logsigma, rotation, flip, layer_indices,
    latent_ref_dict, recon_ref_numpy, device, compute_all_layers, save_latents,
    img_cpu_numpy
):
    """Compute equivariance metrics for one transformation pass.

    Returns a dict ready to be stored in results[key].
    """
    all_layer_metrics = {}

    for layer_idx in layer_indices:
        latent = to_tensor(features[layer_idx]).float().cpu().detach()
        latent_inv = inverse_transformation(latent, rotation, flip)
        reference = latent_ref_dict[layer_idx]
        layer_metrics = compute_equivariance_metrics(latent_inv, reference)

        prefix = f'layer_{layer_idx}_' if compute_all_layers else ''
        for metric_name, metric_val in layer_metrics.items():
            all_layer_metrics[f'{prefix}{metric_name}'] = metric_val

        if save_latents:
            all_layer_metrics[f'layer_{layer_idx}_latent'] = latent_inv.numpy()

    # Reconstruction quality: D(E(T(x))) vs T(x) (cropped to match)
    mi_cpu = mi.cpu().float()
    img_tensor = torch.from_numpy(img_cpu_numpy).float()
    # Crop input to match reconstruction size (model crops by 3:-4)
    _, _, rH, rW = mi_cpu.shape
    _, _, iH, iW = img_tensor.shape
    crop_top = (iH - rH) // 2
    crop_left = (iW - rW) // 2
    img_cropped = img_tensor[:, :, crop_top:crop_top+rH, crop_left:crop_left+rW]
    all_layer_metrics['recon_mse'] = ((mi_cpu - img_cropped) ** 2).mean(dim=[1, 2, 3]).numpy()
    all_layer_metrics['recon_mae'] = (mi_cpu - img_cropped).abs().mean(dim=[1, 2, 3]).numpy()

    # Reconstruction consistency: T(D(E(x_ref))) vs D(E(T(x)))
    mi_ref_tensor = torch.from_numpy(recon_ref_numpy).to(device)
    mi_ref_transformed = apply_transformation(mi_ref_tensor, rotation, flip, device)
    recon_consistency = ((mi_cpu - mi_ref_transformed.cpu().float()) ** 2).mean(dim=[1, 2, 3])
    all_layer_metrics['recon_consistency_mse'] = recon_consistency.numpy()

    result = {
        'reconstruction': mi_cpu.detach().numpy(),
        'logsigma': logsigma.cpu().detach().float().numpy(),
        'img': img_cpu_numpy,
        **all_layer_metrics,
    }
    return result


def evaluate_single_batch(
    model: torch.nn.Module,
    org_image: torch.Tensor,
    channel_ids: torch.Tensor,
    device: torch.device,
    rotations: List[Tuple[float, bool]] = None,
    layer_indices: List[int] = None,
    save_latents: bool = False,
    cont_rotation_modes: List[str] = None
) -> Dict:
    """
    Evaluate equivariance for a single batch with various rotations.

    Methodology:
    - Exact rotations (90° multiples): use torch.rot90, no interpolation artifacts
    - Non-exact rotations: configurable via cont_rotation_modes:
        - 'bilinear': standard bilinear interpolation, no input masking
        - 'circle': mask input to largest inscribed circle before rotation
    - Compute metrics on both full latent and central circular region
    """
    if layer_indices is None:
        layer_indices = [-1]
    if rotations is None:
        rotations = [
            (0, False),
            (90, False),
            (180, False),
            (270, False),
            (30, False),
            (45, False),
            (135, False),
            (0, True),
            (90, True),
        ]
    if cont_rotation_modes is None:
        cont_rotation_modes = ['bilinear', 'circle']

    results = {}
    org_image = org_image.to(torch.float32).to(device)
    batch_size = org_image.shape[0]
    num_channels = org_image.shape[1]
    compute_all_layers = len(layer_indices) > 1 or save_latents

    channel_ids_batch = torch.arange(num_channels).unsqueeze(0).expand(batch_size, -1).to(device)

    autocast_context = cuda_autocast() if (HAS_AUTOCAST and 'cuda' in str(device)) else nullcontext()

    # --- Reference pass: unmasked original ---
    features_org, mi_org, logsigma_org = _run_model_pass(
        model, org_image, channel_ids_batch, autocast_context
    )

    latent_org_dict = {}
    for layer_idx in layer_indices:
        latent_org_dict[layer_idx] = to_tensor(features_org[layer_idx]).float().cpu().detach()

    # Compute reference reconstruction quality
    mi_org_cpu = mi_org.cpu().detach().float()
    org_cpu = org_image.cpu().detach().float()
    _, _, rH, rW = mi_org_cpu.shape
    _, _, iH, iW = org_cpu.shape
    crop_top = (iH - rH) // 2
    crop_left = (iW - rW) // 2
    org_cropped = org_cpu[:, :, crop_top:crop_top+rH, crop_left:crop_left+rW]

    results['r0_nf'] = {
        'reconstruction': mi_org_cpu.numpy(),
        'logsigma': logsigma_org.cpu().detach().float().numpy(),
        'img': org_cpu.numpy(),
        'recon_mse': ((mi_org_cpu - org_cropped) ** 2).mean(dim=[1, 2, 3]).numpy(),
        'recon_mae': (mi_org_cpu - org_cropped).abs().mean(dim=[1, 2, 3]).numpy(),
    }
    if compute_all_layers:
        for layer_idx in layer_indices:
            results['r0_nf'][f'layer_{layer_idx}_latent'] = latent_org_dict[layer_idx].numpy()

    # --- Circle-masked reference pass (only if 'circle' mode is enabled) ---
    latent_circle_dict = None
    if 'circle' in cont_rotation_modes:
        _, _, H, W = org_image.shape
        circle_mask = input_circle_mask(H, W, device)
        org_masked = org_image * circle_mask

        features_circle, mi_circle, logsigma_circle = _run_model_pass(
            model, org_masked, channel_ids_batch, autocast_context
        )

        latent_circle_dict = {}
        for layer_idx in layer_indices:
            latent_circle_dict[layer_idx] = to_tensor(features_circle[layer_idx]).float().cpu().detach()

        mi_circle_cpu = mi_circle.cpu().detach().float()
        org_masked_cpu = org_masked.cpu().detach().float()
        org_masked_cropped = org_masked_cpu[:, :, crop_top:crop_top+rH, crop_left:crop_left+rW]

        results['r0_nf_circle'] = {
            'reconstruction': mi_circle_cpu.numpy(),
            'logsigma': logsigma_circle.cpu().detach().float().numpy(),
            'img': org_masked_cpu.numpy(),
            'recon_mse': ((mi_circle_cpu - org_masked_cropped) ** 2).mean(dim=[1, 2, 3]).numpy(),
            'recon_mae': (mi_circle_cpu - org_masked_cropped).abs().mean(dim=[1, 2, 3]).numpy(),
        }
        if compute_all_layers:
            for layer_idx in layer_indices:
                results['r0_nf_circle'][f'layer_{layer_idx}_latent'] = latent_circle_dict[layer_idx].numpy()

    # --- Transformation passes ---
    for rotation, flip in rotations:
        if rotation == 0 and not flip:
            continue

        is_exact = (rotation % 90 == 0)

        if is_exact:
            # Exact rotations: always use bilinear (no interpolation involved anyway)
            img = apply_transformation(org_image, rotation, flip, device)
            key = f"r{int(rotation)}" + ("_f" if flip else "_nf")

            features, mi, logsigma = _run_model_pass(
                model, img, channel_ids_batch, autocast_context
            )
            results[key] = _compute_transform_metrics(
                features, mi, logsigma, rotation, flip, layer_indices,
                latent_org_dict, results['r0_nf']['reconstruction'],
                device, compute_all_layers, save_latents,
                img.cpu().detach().numpy()
            )
        else:
            # Continuous rotations: process each enabled mode
            if 'bilinear' in cont_rotation_modes:
                img = apply_transformation(org_image, rotation, flip, device)
                key = f"r{int(rotation)}" + ("_f" if flip else "_nf")

                features, mi, logsigma = _run_model_pass(
                    model, img, channel_ids_batch, autocast_context
                )
                results[key] = _compute_transform_metrics(
                    features, mi, logsigma, rotation, flip, layer_indices,
                    latent_org_dict, results['r0_nf']['reconstruction'],
                    device, compute_all_layers, save_latents,
                    img.cpu().detach().numpy()
                )

            if 'circle' in cont_rotation_modes and latent_circle_dict is not None:
                # Mask input to inscribed circle, then rotate
                img_circle = apply_transformation(org_masked, rotation, flip, device)
                key_circle = f"r{int(rotation)}" + ("_f" if flip else "_nf") + "_circle"

                features_c, mi_c, logsigma_c = _run_model_pass(
                    model, img_circle, channel_ids_batch, autocast_context
                )
                results[key_circle] = _compute_transform_metrics(
                    features_c, mi_c, logsigma_c, rotation, flip, layer_indices,
                    latent_circle_dict, results['r0_nf_circle']['reconstruction'],
                    device, compute_all_layers, save_latents,
                    img_circle.cpu().detach().numpy()
                )

    return results


def _is_exact_rotation(key: str) -> bool:
    """Check if a result key corresponds to an exact (90°-multiple or flip) transformation."""
    exact = ['r90_nf', 'r180_nf', 'r270_nf', 'r0_f', 'r90_f']
    return key in exact


def _is_approx_rotation(key: str) -> bool:
    """Check if a result key corresponds to a non-discrete rotation (bilinear mode)."""
    approx = ['r30_nf', 'r45_nf', 'r135_nf']
    return key in approx


def _is_approx_circle(key: str) -> bool:
    """Check if a result key corresponds to a non-discrete rotation (circle mode)."""
    return key.endswith('_circle') and not key.startswith('r0_')


def aggregate_results(all_results: List[Dict]) -> Dict:
    """
    Aggregate results across all batches, separating exact and approximate rotations.
    """
    aggregated = {}

    all_keys = set()
    for results in all_results:
        all_keys.update(results.keys())

    # Include r0_nf and r0_nf_circle in aggregation for recon metrics
    for key in sorted(all_keys):
        if key in ['img_paths', 'panel_idx']:
            continue

        collected = {name: [] for name in METRIC_NAMES}

        for results in all_results:
            if key not in results:
                continue
            data = results[key]

            for name in METRIC_NAMES:
                if name in data:
                    collected[name].append(data[name])
                else:
                    for dk in data.keys():
                        if dk.endswith(f'_{name}'):
                            collected[name].append(data[dk])
                            break

        stats = {}
        for name in METRIC_NAMES:
            if collected[name]:
                vals = np.concatenate(collected[name])
                stats[f'{name}_mean'] = float(vals.mean())
                stats[f'{name}_std'] = float(vals.std())
                stats[f'{name}_median'] = float(np.median(vals))

        if stats:
            aggregated[key] = stats

    # Add summary groups: exact vs approximate rotations
    exact_keys = [k for k in aggregated if _is_exact_rotation(k)]
    approx_keys = [k for k in aggregated if _is_approx_rotation(k)]
    approx_circle_keys = [k for k in aggregated if _is_approx_circle(k)]

    groups = [('EXACT_MEAN', exact_keys), ('APPROX_MEAN', approx_keys)]
    if approx_circle_keys:
        groups.append(('APPROX_CIRCLE_MEAN', approx_circle_keys))

    for group_name, group_keys in groups:
        if not group_keys:
            continue
        group_stats = {}
        for name in METRIC_NAMES:
            mean_key = f'{name}_mean'
            vals = [aggregated[k][mean_key] for k in group_keys if mean_key in aggregated[k]]
            if vals:
                group_stats[mean_key] = float(np.mean(vals))
        if group_stats:
            aggregated[group_name] = group_stats

    return aggregated


METRIC_NAMES = [
    'recon_mse', 'recon_mae',
    'equiv_mse', 'relative_mse', 'equiv_l1',
    'cosine_sim', 'pearson', 'channelwise_pearson',
    'central_mse', 'central_relative_mse', 'central_pearson',
    'recon_consistency_mse',
]

METRIC_LABELS = {
    'recon_mse': 'Reconstruction MSE',
    'recon_mae': 'Reconstruction MAE',
    'equiv_mse': 'Equivariance MSE',
    'relative_mse': 'Relative Equiv. MSE',
    'equiv_l1': 'Equivariance L1',
    'cosine_sim': 'Cosine Similarity',
    'pearson': 'Pearson Correlation',
    'channelwise_pearson': 'Channelwise Pearson',
    'central_mse': 'Central Equiv. MSE',
    'central_relative_mse': 'Central Relative MSE',
    'central_pearson': 'Central Pearson',
    'recon_consistency_mse': 'Recon. Consistency MSE',
}


# ============================================================
# Plotting functions
# ============================================================

def _sanitize_filename(s: str) -> str:
    """Sanitize a string for use as a filename."""
    return s.replace('/', '_').replace('\\', '_').replace(':', '_').replace(' ', '_')[:80]


def plot_reconstruction_grid_sample(results, sample_idx, transform_key, save_path, ncols=9):
    """Plot reconstruction grid (orig, reconstructed, uncertainty) for all channels."""
    data = results[transform_key]
    orig = data['img'][sample_idx]          # [C, H, W]
    recon = data['reconstruction'][sample_idx]  # [C, H, W]
    logsigma = data.get('logsigma', np.zeros_like(data['reconstruction']))[sample_idx]
    sigma = np.exp(logsigma)

    num_channels = orig.shape[0]
    nrows = ceil(num_channels / (ncols // 3))
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    ax_flat = axs.flatten() if nrows > 1 else (list(axs) if ncols > 1 else [axs])

    for ax in ax_flat:
        ax.axis('off')

    for i in range(0, len(ax_flat), 3):
        j = i // 3
        if j >= num_channels:
            break
        ax_flat[i].imshow(orig[j], cmap='CMRmap', vmin=0, vmax=1)
        ax_flat[i].set_title(f'Orig Ch{j}', fontsize=8)
        ax_flat[i + 1].imshow(recon[j], cmap='CMRmap', vmin=0, vmax=1)
        ax_flat[i + 1].set_title(f'Recon Ch{j}', fontsize=8)
        ax_flat[i + 2].imshow(sigma[j], cmap='CMRmap')
        ax_flat[i + 2].set_title(f'Var Ch{j}', fontsize=8)

    fig.suptitle(f'Reconstruction Grid ({transform_key})', fontsize=12, fontweight='bold')
    fig.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_reconstruction_comparison_sample(results, sample_idx, save_path):
    """Plot reconstruction comparison across all transformations, using CMRmap vmin=0, vmax=1."""
    # Collect all transform keys (exclude metadata keys)
    transform_keys = [k for k in results if k not in ('img_paths', 'panel_idx')
                      and isinstance(results[k], dict) and 'reconstruction' in results[k]]
    transform_keys = sorted(transform_keys)

    if not transform_keys:
        return

    ref_data = results['r0_nf']
    num_channels = ref_data['img'][sample_idx].shape[0]
    # Pick first 3 channels for display
    display_channels = list(range(min(3, num_channels)))

    ncols = len(transform_keys)
    nrows = len(display_channels) * 2  # orig row + recon row per channel

    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    if nrows == 1:
        axs = axs[np.newaxis, :]
    if ncols == 1:
        axs = axs[:, np.newaxis]

    for col_idx, tkey in enumerate(transform_keys):
        data = results[tkey]
        img = data['img'][sample_idx]
        recon = data['reconstruction'][sample_idx]

        for ch_i, ch in enumerate(display_channels):
            row_img = ch_i * 2
            row_rec = ch_i * 2 + 1

            axs[row_img, col_idx].imshow(img[ch], cmap='CMRmap', vmin=0, vmax=1)
            axs[row_img, col_idx].axis('off')
            if col_idx == 0:
                axs[row_img, col_idx].set_ylabel(f'Input Ch{ch}', fontsize=8)

            axs[row_rec, col_idx].imshow(recon[ch], cmap='CMRmap', vmin=0, vmax=1)
            axs[row_rec, col_idx].axis('off')
            if col_idx == 0:
                axs[row_rec, col_idx].set_ylabel(f'Recon Ch{ch}', fontsize=8)

        axs[0, col_idx].set_title(tkey, fontsize=8)

    fig.suptitle('Reconstruction Comparison', fontsize=12, fontweight='bold')
    fig.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_latent_grid_sample(results, sample_idx, transform_key, save_path, layer_idx=-1, ncols=8):
    """Plot latent feature maps for all channels in a grid, using CMRmap colormap."""
    data = results[transform_key]
    latent_key = f'layer_{layer_idx}_latent'
    if latent_key not in data:
        print(f"Warning: {latent_key} not found in {transform_key}, skipping latent grid plot")
        return

    latent = data[latent_key][sample_idx]  # [C, H, W]
    num_channels = latent.shape[0]

    if num_channels == 0:
        print(f"Warning: latent has 0 channels for {transform_key}, skipping latent grid plot")
        return
    nrows = ceil(num_channels / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    ax_flat = axs.flatten() if nrows > 1 else (list(axs) if ncols > 1 else [axs])

    for ax in ax_flat:
        ax.axis('off')

    for j in range(min(num_channels, len(ax_flat))):
        ax_flat[j].imshow(latent[j], cmap='CMRmap')
        ax_flat[j].set_title(f'Ch{j}', fontsize=7)

    fig.suptitle(f'Latent Grid Layer {layer_idx} ({transform_key})', fontsize=12, fontweight='bold')
    fig.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_latent_comparison_sample(results, sample_idx, transform_key, save_path, layer_idx=-1, ncols=6):
    """Plot latent comparison: reference, transformed, relative error per channel."""
    ref_key = 'r0_nf_circle' if transform_key.endswith('_circle') else 'r0_nf'
    ref_data = results.get(ref_key, results.get('r0_nf'))
    trans_data = results[transform_key]

    ref_latent_key = f'layer_{layer_idx}_latent'
    if ref_latent_key not in ref_data:
        print(f"Warning: {ref_latent_key} not found in {ref_key} for {transform_key}, skipping latent comparison plot")
        return
    if ref_latent_key not in trans_data:
        print(f"Warning: {ref_latent_key} not found in {transform_key}, skipping latent comparison plot")
        return

    latent_ref = ref_data[ref_latent_key][sample_idx]    # [C, H, W]
    latent_trans = trans_data[ref_latent_key][sample_idx]  # [C, H, W]
    num_channels = latent_ref.shape[0]

    if num_channels == 0:
        print(f"Warning: latent has 0 channels for {transform_key}, skipping latent comparison plot")
        return

    nrows = ceil(num_channels / (ncols // 3))
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    if nrows == 1:
        ax_flat = list(axs) if ncols > 1 else [axs]
    else:
        ax_flat = axs.flatten()

    for ax in ax_flat:
        ax.axis('off')

    for i in range(0, len(ax_flat), 3):
        j = i // 3
        if j >= num_channels:
            break

        vmin = min(latent_ref[j].min(), latent_trans[j].min())
        vmax = max(latent_ref[j].max(), latent_trans[j].max())

        ax_flat[i].imshow(latent_ref[j], cmap='CMRmap', vmin=vmin, vmax=vmax)
        ax_flat[i].set_title(f'Ref Ch{j}', fontsize=7)

        ax_flat[i + 1].imshow(latent_trans[j], cmap='CMRmap', vmin=vmin, vmax=vmax)
        ax_flat[i + 1].set_title(f'Trans Ch{j}', fontsize=7)

        # Relative error: |diff|^2 / (ref^2 + eps)
        ref_energy = latent_ref[j] ** 2 + 1e-8
        rel_error = (latent_ref[j] - latent_trans[j]) ** 2 / ref_energy
        ax_flat[i + 2].imshow(rel_error, cmap='hot')
        ax_flat[i + 2].set_title(f'RelErr Ch{j}', fontsize=7)

    fig.suptitle(f'Latent Comparison ({transform_key}, Layer {layer_idx})', fontsize=12, fontweight='bold')
    fig.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_metrics_bar(aggregated_metrics, metric_name, save_path):
    """Bar plot for one metric across all transformations."""
    keys = []
    means = []
    stds = []
    mean_key = f'{metric_name}_mean'
    std_key = f'{metric_name}_std'

    for k in sorted(aggregated_metrics.keys()):
        if k.startswith(('EXACT_MEAN', 'APPROX_MEAN', 'APPROX_CIRCLE_MEAN')):
            continue
        if mean_key in aggregated_metrics[k]:
            keys.append(k)
            means.append(aggregated_metrics[k][mean_key])
            stds.append(aggregated_metrics[k].get(std_key, 0))

    if not keys:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 0.8), 5))
    x = np.arange(len(keys))
    ax.bar(x, means, yerr=stds, capsize=4, alpha=0.7, color='steelblue')
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(METRIC_LABELS.get(metric_name, metric_name))
    ax.set_title(METRIC_LABELS.get(metric_name, metric_name), fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_sample_plots(results, sample_idx, sample_dir, layer_indices):
    """Save all plots for a single sample into sample_dir."""
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    transform_keys = [k for k in results if k not in ('img_paths', 'panel_idx')
                      and isinstance(results[k], dict)]

    # Reconstruction grid for each transform
    for tkey in transform_keys:
        if 'reconstruction' in results[tkey]:
            plot_reconstruction_grid_sample(
                results, sample_idx, tkey,
                str(sample_dir / f'reconstruction_grid_{tkey}.png')
            )

    # Reconstruction comparison (all transforms side by side)
    if any('reconstruction' in results[k] for k in transform_keys):
        plot_reconstruction_comparison_sample(
            results, sample_idx,
            str(sample_dir / 'reconstruction_comparison.png')
        )

    # Latent grids and comparisons
    for layer_idx in layer_indices:
        for tkey in transform_keys:
            latent_key = f'layer_{layer_idx}_latent'
            if latent_key in results[tkey]:
                plot_latent_grid_sample(
                    results, sample_idx, tkey,
                    str(sample_dir / f'latent_grid_{tkey}_layer{layer_idx}.png'),
                    layer_idx=layer_idx
                )
                # Comparison only for non-identity transforms
                if tkey not in ('r0_nf', 'r0_nf_circle'):
                    plot_latent_comparison_sample(
                        results, sample_idx, tkey,
                        str(sample_dir / f'latent_comparison_{tkey}_layer{layer_idx}.png'),
                        layer_idx=layer_idx
                    )


def save_metrics_csv(all_results, output_path):
    """Save per-sample metrics to a CSV file."""
    rows = []
    for batch_idx, results in enumerate(all_results):
        img_paths = results.get('img_paths', [])
        panel_idxs = results.get('panel_idx', [])

        transform_keys = [k for k in results if k not in ('img_paths', 'panel_idx')
                          and isinstance(results[k], dict)]

        batch_size = 0
        for k in transform_keys:
            for mn in METRIC_NAMES:
                if mn in results[k]:
                    batch_size = len(results[k][mn])
                    break
            if batch_size > 0:
                break

        for s in range(batch_size):
            sample_id = img_paths[s] if s < len(img_paths) else f'batch{batch_idx}_s{s}'
            panel = panel_idxs[s] if hasattr(panel_idxs, '__getitem__') and s < len(panel_idxs) else ''

            for tkey in sorted(transform_keys):
                data = results[tkey]
                row = {
                    'sample_id': sample_id,
                    'panel_idx': panel,
                    'batch_idx': batch_idx,
                    'sample_in_batch': s,
                    'transformation': tkey,
                }
                for mn in METRIC_NAMES:
                    # Handle both unprefixed and layer-prefixed metric names
                    if mn in data and s < len(data[mn]):
                        row[mn] = float(data[mn][s])
                    else:
                        # Look for prefixed version (e.g., layer_-1_equiv_mse)
                        found = False
                        for dk in data.keys():
                            if dk.endswith(f'_{mn}') and s < len(data[dk]):
                                row[mn] = float(data[dk][s])
                                found = True
                                break
                        if not found:
                            row[mn] = ''
                rows.append(row)

    if not rows:
        return

    fieldnames = ['sample_id', 'panel_idx', 'batch_idx', 'sample_in_batch', 'transformation'] + METRIC_NAMES
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Metrics CSV saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate rotation equivariance (v2)')
    parser.add_argument(
        '--config',
        type=str,
        default='train_masked_equivariant_config_flip_v2.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (overrides config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for evaluation'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='equivariance_results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--num-batches',
        type=int,
        default=None,
        help='Number of batches to evaluate (None = all)'
    )
    parser.add_argument(
        '--save-features',
        action='store_true',
        help='Save intermediate features from all layers'
    )
    parser.add_argument(
        '--save-reconstructions',
        action='store_true',
        help='Save reconstructions and images'
    )
    parser.add_argument(
        '--layer-indices',
        type=int,
        nargs='+',
        default=[-1],
        help='Layer indices to evaluate (default: -1 for last layer only)'
    )
    parser.add_argument(
        '--save-latents',
        action='store_true',
        help='Save latent representations for visualization'
    )
    parser.add_argument(
        '--cont-rotation-modes',
        nargs='+',
        default=['bilinear', 'circle'],
        choices=['bilinear', 'circle'],
        help='Processing modes for continuous (non-90) rotations. '
             '"bilinear": standard bilinear interpolation (no input masking). '
             '"circle": mask input to largest inscribed circle before rotation. '
             'Default: both.'
    )
    parser.add_argument(
        '--fraction',
        type=float,
        default=1.0,
        help='Fraction of the dataset to evaluate (0.0-1.0). Default: 1.0 (all)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for dataset subsampling when --fraction < 1.0'
    )
    parser.add_argument(
        '--num-plot-samples',
        type=int,
        default=1,
        help='Number of samples to generate detailed plots for (default: 1)'
    )

    args = parser.parse_args()

    print(f"Loading model from config: {args.config}")
    print(f"Using device: {args.device}")

    model, train_dataloader, test_dataloader, TOKENIZER, INV_TOKENIZER, config = load_model_and_data(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    # Build output dir with checkpoint name
    ckpt_path = args.checkpoint or config.get('from_checkpoint', '')
    ckpt_name = os.path.basename(ckpt_path).replace("checkpoint-", "").replace(".pth", "")
    output_dir = Path(args.output_dir) / ckpt_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Force save_latents when we need plots (latent plots require saved latents)
    save_latents = args.save_latents or args.num_plot_samples > 0

    print("\n" + "="*80)
    print("Starting Rotation Equivariance Evaluation (v2)")
    print(f"Continuous rotation modes: {args.cont_rotation_modes}")
    if args.fraction < 1.0:
        print(f"Processing fraction: {args.fraction:.2%} of dataset (seed={args.seed})")
    if args.num_plot_samples > 0:
        print(f"Will generate plots for {args.num_plot_samples} sample(s)")
    print("="*80)

    # Dataset subsampling
    total_batches = len(test_dataloader)
    if args.num_batches:
        total_batches = min(args.num_batches, total_batches)

    if args.fraction < 1.0:
        num_to_process = max(1, int(total_batches * args.fraction))
        rng = np.random.RandomState(args.seed)
        selected_indices = set(rng.choice(total_batches, num_to_process, replace=False))
        print(f"Selected {num_to_process} / {total_batches} batches")
    else:
        num_to_process = total_batches
        selected_indices = None

    # Evaluation loop
    all_results = []
    plots_dir = output_dir / 'sample_plots'
    samples_plotted = 0

    with torch.no_grad():
        for i, (org_image, channel_ids, panel_idx, img_path) in enumerate(tqdm(
            test_dataloader,
            total=total_batches,
            desc="Evaluating"
        )):
            if i >= total_batches:
                break
            if selected_indices is not None and i not in selected_indices:
                continue

            results = evaluate_single_batch(
                model=model,
                org_image=org_image,
                channel_ids=channel_ids,
                device=args.device if args.device != 'cpu' else model.parameters().__next__().device,
                layer_indices=args.layer_indices,
                save_latents=save_latents,
                cont_rotation_modes=args.cont_rotation_modes
            )

            results['img_paths'] = img_path
            if isinstance(panel_idx, torch.Tensor):
                results['panel_idx'] = panel_idx.numpy()
            else:
                results['panel_idx'] = panel_idx

            # Generate per-sample plots for the first N samples
            if samples_plotted < args.num_plot_samples:
                batch_size = org_image.shape[0]
                for s in range(batch_size):
                    if samples_plotted >= args.num_plot_samples:
                        break
                    sample_name = _sanitize_filename(
                        img_path[s] if s < len(img_path) else f'batch{i}_s{s}'
                    )
                    sample_dir = plots_dir / sample_name
                    save_sample_plots(results, s, sample_dir, args.layer_indices)
                    samples_plotted += 1

            all_results.append(results)

    print("\n" + "="*80)
    print("Computing Aggregate Metrics")
    print("="*80)

    aggregated_metrics = aggregate_results(all_results)

    # Print results
    print("\nEquivariance Metrics:")
    print("=" * 100)

    for key, metrics in sorted(aggregated_metrics.items()):
        if key in ['EXACT_MEAN', 'APPROX_MEAN', 'APPROX_CIRCLE_MEAN']:
            continue
        print(f"\n{key}:")
        print(f"  Recon MSE:         {metrics.get('recon_mse_mean', float('nan')):.6f} +/- {metrics.get('recon_mse_std', float('nan')):.6f}")
        print(f"  Recon MAE:         {metrics.get('recon_mae_mean', float('nan')):.6f} +/- {metrics.get('recon_mae_std', float('nan')):.6f}")
        print(f"  Equiv Rel MSE:     {metrics.get('relative_mse_mean', float('nan')):.6f} +/- {metrics.get('relative_mse_std', float('nan')):.6f}")
        print(f"  Equiv Raw MSE:     {metrics.get('equiv_mse_mean', float('nan')):.6f} +/- {metrics.get('equiv_mse_std', float('nan')):.6f}")
        print(f"  Pearson:           {metrics.get('pearson_mean', float('nan')):.6f} +/- {metrics.get('pearson_std', float('nan')):.6f}")
        print(f"  CW Pearson:        {metrics.get('channelwise_pearson_mean', float('nan')):.6f}")
        print(f"  Central Rel MSE:   {metrics.get('central_relative_mse_mean', float('nan')):.6f}")
        print(f"  Central Pearson:   {metrics.get('central_pearson_mean', float('nan')):.6f}")
        print(f"  Recon Consistency: {metrics.get('recon_consistency_mse_mean', float('nan')):.6f}")

    # Print group summaries
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for group in ['r0_nf', 'r0_nf_circle', 'EXACT_MEAN', 'APPROX_MEAN', 'APPROX_CIRCLE_MEAN']:
        if group not in aggregated_metrics:
            continue
        m = aggregated_metrics[group]
        labels = {
            'r0_nf': "Reference (identity)",
            'r0_nf_circle': "Reference (circle-masked)",
            'EXACT_MEAN': "Exact rotations (90 multiples + flips)",
            'APPROX_MEAN': "Approximate rotations - bilinear (30, 45, 135)",
            'APPROX_CIRCLE_MEAN': "Approximate rotations - circle masked (30, 45, 135)",
        }
        print(f"\n{labels[group]}:")
        print(f"  Recon MSE:         {m.get('recon_mse_mean', float('nan')):.6f}")
        print(f"  Recon MAE:         {m.get('recon_mae_mean', float('nan')):.6f}")
        print(f"  Equiv Rel MSE:     {m.get('relative_mse_mean', float('nan')):.6f}")
        print(f"  Pearson:           {m.get('pearson_mean', float('nan')):.6f}")
        print(f"  Central Rel MSE:   {m.get('central_relative_mse_mean', float('nan')):.6f}")
        print(f"  Central Pearson:   {m.get('central_pearson_mean', float('nan')):.6f}")
        print(f"  Recon Consistency: {m.get('recon_consistency_mse_mean', float('nan')):.6f}")

    # Save aggregated metrics
    metrics_path = output_dir / 'aggregated_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(aggregated_metrics, f, indent=2)
    print(f"\nAggregated metrics saved to: {metrics_path}")

    # Save per-sample metrics CSV
    csv_path = output_dir / 'metrics.csv'
    save_metrics_csv(all_results, csv_path)

    # Save per-metric bar plots
    metrics_plots_dir = output_dir / 'metric_plots'
    metrics_plots_dir.mkdir(parents=True, exist_ok=True)
    for metric_name in METRIC_NAMES:
        plot_metrics_bar(
            aggregated_metrics, metric_name,
            str(metrics_plots_dir / f'{metric_name}.png')
        )
    print(f"Metric plots saved to: {metrics_plots_dir}")

    if samples_plotted > 0:
        print(f"Sample plots saved to: {plots_dir} ({samples_plotted} samples)")

    # Save full results if explicitly requested (not just from auto-enabled save_latents for plots)
    if args.save_reconstructions or args.save_features or args.save_latents:
        print("\nSaving detailed results...")

        for i, results in enumerate(tqdm(all_results, desc="Saving")):
            batch_file = output_dir / f'batch_{i:04d}.npz'

            save_dict = {}

            for key, value in results.items():
                if key in ['img_paths', 'panel_idx']:
                    save_dict[key] = value
                    continue

                if args.save_reconstructions:
                    if 'img' in value:
                        save_dict[f'{key}_img'] = value['img']
                    if 'reconstruction' in value:
                        save_dict[f'{key}_reconstruction'] = value['reconstruction']
                    if 'logsigma' in value:
                        save_dict[f'{key}_logsigma'] = value['logsigma']

                if args.save_features and 'all_features' in value:
                    for layer_idx, features in enumerate(value['all_features']):
                        save_dict[f'{key}_layer_{layer_idx}'] = features

                if save_latents:
                    for sub_key in value:
                        if '_latent' in sub_key:
                            save_dict[f'{key}_{sub_key}'] = value[sub_key]

                # Save all metrics
                for metric_key in METRIC_NAMES:
                    if metric_key in value:
                        save_dict[f'{key}_{metric_key}'] = value[metric_key]

            np.savez_compressed(batch_file, **save_dict)

        print(f"Detailed results saved to: {output_dir}")

    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)

    return aggregated_metrics, all_results


if __name__ == '__main__':
    main()
