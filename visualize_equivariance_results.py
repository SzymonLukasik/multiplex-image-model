"""
Visualization utilities for equivariance evaluation results.

Provides functions to:
- Plot equivariance metrics
- Visualize latent representations
- Compare original vs transformed latents
- Generate reconstruction comparisons
"""

import argparse
import json
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from matplotlib.gridspec import GridSpec


def load_results(results_dir: str) -> Tuple[Dict, List[Dict]]:
    """
    Load evaluation results from directory.

    Args:
        results_dir: Directory containing results

    Returns:
        Tuple of (aggregated_metrics, batch_results_list)
    """
    results_path = Path(results_dir)

    # Load aggregated metrics
    metrics_path = results_path / 'aggregated_metrics.json'
    with open(metrics_path, 'r') as f:
        aggregated_metrics = json.load(f)

    # Load batch results
    batch_files = sorted(results_path.glob('batch_*.npz'))
    batch_results = []

    for batch_file in batch_files:
        data = np.load(batch_file, allow_pickle=True)
        batch_results.append(dict(data))

    return aggregated_metrics, batch_results


def plot_metrics_comparison(
    aggregated_metrics: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Plot comparison of equivariance metrics across transformations.

    Args:
        aggregated_metrics: Dictionary with aggregated metrics
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    # Extract data
    transformations = []
    mse_means = []
    mse_stds = []
    cosine_means = []
    cosine_stds = []

    for key, metrics in sorted(aggregated_metrics.items()):
        transformations.append(key)
        mse_means.append(metrics['equiv_mse_mean'])
        mse_stds.append(metrics['equiv_mse_std'])
        cosine_means.append(metrics['cosine_sim_mean'])
        cosine_stds.append(metrics['cosine_sim_std'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot MSE
    x = np.arange(len(transformations))
    ax1.bar(x, mse_means, yerr=mse_stds, capsize=5, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Transformation', fontsize=12)
    ax1.set_ylabel('Equivariance Error (MSE)', fontsize=12)
    ax1.set_title('Rotation Equivariance Error (MSE)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(transformations, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    # Plot Cosine Similarity
    ax2.bar(x, cosine_means, yerr=cosine_stds, capsize=5, alpha=0.7, color='coral')
    ax2.set_xlabel('Transformation', fontsize=12)
    ax2.set_ylabel('Cosine Similarity', fontsize=12)
    ax2.set_title('Cosine Similarity Between Latents', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(transformations, rotation=45, ha='right')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect similarity')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def rotate_back_tensor(x: torch.Tensor, rotation: float) -> torch.Tensor:
    """Rotate tensor back to original orientation."""
    if rotation % 90 == 0:
        k = int(-rotation // 90) % 4
        return torch.rot90(x, k=k, dims=(-2, -1))
    else:
        return TF.rotate(
            x, -rotation,
            interpolation=TF.InterpolationMode.BILINEAR,
            expand=False,
            center=((x.shape[-1] - 1) / 2, (x.shape[-2] - 1) / 2)
        )


def plot_latent_comparison(
    batch_data: Dict,
    batch_idx: int = 0,
    rotation: int = 90,
    flip: bool = False,
    channel_idx: int = 0,
    layer_idx: int = -1,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 6)
):
    """
    Compare original and transformed latent representations.

    Args:
        batch_data: Dictionary with batch results
        batch_idx: Index within batch
        rotation: Rotation angle
        flip: Whether flip was applied
        channel_idx: Channel index to visualize
        layer_idx: Layer index (-1 for final layer)
        save_path: Path to save figure
        figsize: Figure size
    """
    # Construct keys
    key_transformed = f"r{rotation}" + ("_f" if flip else "_nf")
    use_mask = rotation % 90 != 0
    key_reference = "r0_nf" + ("_mask" if use_mask else "")

    # Load latents - check if features were saved
    ref_key = f'{key_reference}_layer_{layer_idx}'
    trans_key = f'{key_transformed}_layer_{layer_idx}'

    if ref_key not in batch_data or trans_key not in batch_data:
        print(f"Note: Latent features not saved. Use --layer-indices with 2+ layers or run evaluation with --save-features to enable latent visualization.")
        return

    latent_ref = batch_data[ref_key][batch_idx, channel_idx]  # [H, W]
    latent_trans = batch_data[trans_key][batch_idx, channel_idx]  # [H, W]

    # Compute difference
    diff = np.abs(latent_ref - latent_trans)

    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 4, figure=fig, wspace=0.3)

    # Original latent
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(latent_ref, cmap='viridis')
    ax1.set_title(f'Reference Latent\n(Channel {channel_idx})', fontsize=11)
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Transformed latent
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(latent_trans, cmap='viridis')
    ax2.set_title(f'Transformed Latent\n(r={rotation}°' + (', flip' if flip else '') + ')', fontsize=11)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Difference
    ax3 = fig.add_subplot(gs[2])
    im3 = ax3.imshow(diff, cmap='hot')
    ax3.set_title('Absolute Difference', fontsize=11)
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # Metrics
    ax4 = fig.add_subplot(gs[3])
    ax4.axis('off')

    # Compute metrics
    mse = np.mean(diff ** 2)
    mae = np.mean(diff)
    max_error = np.max(diff)

    # Get batch-level metrics if available
    metric_key_mse = f'{key_transformed}_equiv_mse'
    metric_key_cos = f'{key_transformed}_cosine_sim'

    batch_mse = batch_data.get(metric_key_mse, [None])[batch_idx] if metric_key_mse in batch_data else None
    batch_cos = batch_data.get(metric_key_cos, [None])[batch_idx] if metric_key_cos in batch_data else None

    metrics_text = f"Layer {layer_idx} Metrics:\n"
    metrics_text += f"  Channel {channel_idx}:\n"
    metrics_text += f"    MSE: {mse:.6f}\n"
    metrics_text += f"    MAE: {mae:.6f}\n"
    metrics_text += f"    Max: {max_error:.6f}\n\n"

    if batch_mse is not None:
        metrics_text += f"  Full Latent:\n"
        metrics_text += f"    MSE: {batch_mse:.6f}\n"
    if batch_cos is not None:
        metrics_text += f"    Cosine: {batch_cos:.6f}\n"

    ax4.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Latent comparison saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_reconstruction_comparison(
    batch_data: Dict,
    batch_idx: int = 0,
    rotation: int = 90,
    flip: bool = False,
    channel_idx: int = 0,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 5)
):
    """
    Compare reconstructions across transformations.

    Args:
        batch_data: Dictionary with batch results
        batch_idx: Index within batch
        rotation: Rotation angle
        flip: Whether flip was applied
        channel_idx: Marker channel to visualize
        save_path: Path to save figure
        figsize: Figure size
    """
    # Construct keys
    key_original = "r0_nf"
    key_transformed = f"r{rotation}" + ("_f" if flip else "_nf")

    # Load data
    img_orig = batch_data.get(f'{key_original}_img')
    recon_orig = batch_data.get(f'{key_original}_reconstruction')
    img_trans = batch_data.get(f'{key_transformed}_img')
    recon_trans = batch_data.get(f'{key_transformed}_reconstruction')

    if img_orig is None or recon_orig is None or img_trans is None or recon_trans is None:
        print("Warning: Reconstruction data not found in batch")
        return

    # Extract samples
    img_orig = img_orig[batch_idx, channel_idx]
    recon_orig = recon_orig[batch_idx, channel_idx]
    img_trans = img_trans[batch_idx, channel_idx]
    recon_trans = recon_trans[batch_idx, channel_idx]

    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Original image
    im0 = axes[0].imshow(img_orig, cmap='gray')
    axes[0].set_title(f'Original Image\n(Channel {channel_idx})', fontsize=11)
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Original reconstruction
    im1 = axes[1].imshow(recon_orig, cmap='gray')
    axes[1].set_title('Reconstruction', fontsize=11)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Transformed image
    im2 = axes[2].imshow(img_trans, cmap='gray')
    axes[2].set_title(f'Transformed Image\n(r={rotation}°' + (', flip' if flip else '') + ')', fontsize=11)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Transformed reconstruction
    im3 = axes[3].imshow(recon_trans, cmap='gray')
    axes[3].set_title('Reconstruction', fontsize=11)
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reconstruction comparison saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_equivariance_heatmap(
    batch_results: List[Dict],
    metric: str = 'equiv_mse',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Plot heatmap of equivariance errors across samples and transformations.

    Args:
        batch_results: List of batch result dictionaries
        metric: Metric to visualize ('equiv_mse', 'equiv_l1', 'cosine_sim')
        save_path: Path to save figure
        figsize: Figure size
    """
    # Collect data
    transformations = []
    data_matrix = []

    # Get all transformation keys
    sample_batch = batch_results[0]
    for key in sorted(sample_batch.keys()):
        if metric in key and key.endswith(metric):
            trans_key = key.replace(f'_{metric}', '')
            transformations.append(trans_key)

    # Collect metrics for each transformation
    for trans_key in transformations:
        metric_key = f'{trans_key}_{metric}'
        trans_data = []

        for batch in batch_results:
            if metric_key in batch:
                trans_data.extend(batch[metric_key].flatten())

        if trans_data:  # Only add if we have data
            data_matrix.append(trans_data)

    # Check if we have enough data
    if not data_matrix or not transformations:
        print(f"Note: No data found for metric '{metric}' heatmap. Skipping.")
        return

    # Convert to array
    data_matrix = np.array(data_matrix)  # [num_transformations, num_samples]

    # Check if matrix is empty or has too few samples
    if data_matrix.size == 0 or data_matrix.shape[1] < 2:
        print(f"Note: Insufficient samples ({data_matrix.shape[1] if data_matrix.size > 0 else 0}) for heatmap visualization (need at least 2). Skipping.")
        return

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(data_matrix, aspect='auto', cmap='RdYlGn_r')

    # Set ticks
    ax.set_yticks(np.arange(len(transformations)))
    ax.set_yticklabels(transformations)
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Transformation', fontsize=12)

    metric_title = {
        'equiv_mse': 'Equivariance Error (MSE)',
        'equiv_l1': 'Equivariance Error (L1)',
        'cosine_sim': 'Cosine Similarity'
    }
    ax.set_title(f'{metric_title.get(metric, metric)} Across Samples', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_title.get(metric, metric), fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_reconstruction_grid(
    batch_data: Dict,
    sample_idx: int = 0,
    rotation: int = 0,
    flip: bool = False,
    ncols: int = 9,
    scale_by_max: bool = True,
    save_path: Optional[str] = None,
):
    """
    Plot reconstructions with uncertainty for all channels in a grid layout.

    Similar to plot_reconstructs_with_uncertainty, shows original, reconstructed,
    and uncertainty for each channel.

    Args:
        batch_data: Dictionary with batch results
        sample_idx: Sample index within batch
        rotation: Rotation angle (0, 30, 45, 90, 135, 180, 270)
        flip: Whether flip was applied
        ncols: Number of columns (should be multiple of 3)
        scale_by_max: Whether to scale uncertainty by max value
        save_path: Path to save figure
    """
    # Construct key for this transformation
    key = f"r{rotation}" + ("_f" if flip else "_nf")

    # Get data
    img_key = f'{key}_img'
    recon_key = f'{key}_reconstruction'
    logsigma_key = f'{key}_logsigma'

    if img_key not in batch_data:
        print(f"Warning: No image data found for {key}. Run evaluation with --save-reconstructions.")
        return None

    orig_img = batch_data[img_key][sample_idx]  # [C, H, W]

    if recon_key not in batch_data:
        print(f"Warning: No reconstruction data found for {key}.")
        return None
    reconstructed_img = batch_data[recon_key][sample_idx]  # [C, H, W]

    # Get uncertainty (sigma from logsigma)
    if logsigma_key in batch_data:
        logsigma = batch_data[logsigma_key][sample_idx]
        sigma = np.exp(logsigma)  # Convert logsigma to sigma (variance)
    else:
        sigma = np.zeros_like(reconstructed_img)

    num_channels = orig_img.shape[0]

    # Calculate grid dimensions
    nrows = ceil(num_channels / (ncols // 3))
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    ax_flat = axs.flatten() if nrows > 1 else axs

    # Turn off all axes first
    for ax in ax_flat:
        ax.axis('off')

    for i in range(0, len(ax_flat), 3):
        j = i // 3

        if j >= num_channels:
            break

        ax_img = ax_flat[i]
        ax_reconstructed = ax_flat[i + 1]
        ax_uncertainty = ax_flat[i + 2]

        # Original image
        ax_img.imshow(orig_img[j], cmap='CMRmap', vmin=0, vmax=1)
        ax_img.set_title(f'Original\nCh {j}', fontsize=9)
        ax_img.axis('off')

        # Reconstructed image
        ax_reconstructed.imshow(reconstructed_img[j], cmap='CMRmap', vmin=0, vmax=1)
        ax_reconstructed.set_title(f'Reconstructed\nCh {j}', fontsize=9)
        ax_reconstructed.axis('off')

        # Uncertainty (variance)
        if scale_by_max:
            var_min = sigma[j].min()
            var_max = sigma[j].max()
        else:
            var_min = None
            var_max = None

        ax_uncertainty.imshow(sigma[j], cmap='CMRmap', vmin=var_min, vmax=var_max)
        ax_uncertainty.set_title(f'Variance\nCh {j}', fontsize=9)
        ax_uncertainty.axis('off')

    # Add overall title
    flip_str = " + flip" if flip else ""
    fig.suptitle(f'Reconstruction Grid (rotation={rotation}°{flip_str})', fontsize=12, fontweight='bold')
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reconstruction grid saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_latent_grid(
    batch_data: Dict,
    sample_idx: int = 0,
    rotation: int = 0,
    flip: bool = False,
    layer_idx: int = -1,
    ncols: int = 8,
    save_path: Optional[str] = None,
):
    """
    Plot latent representations (encoder output) for all channels in a grid layout.

    Shows the latent feature maps from the encoder for visualization.

    Args:
        batch_data: Dictionary with batch results
        sample_idx: Sample index within batch
        rotation: Rotation angle (0, 30, 45, 90, 135, 180, 270)
        flip: Whether flip was applied
        layer_idx: Layer index for latent (-1 for last layer)
        ncols: Number of columns in the grid
        save_path: Path to save figure
    """
    # Construct key for this transformation
    key = f"r{rotation}" + ("_f" if flip else "_nf")

    # Try different key formats for latents
    latent_key = f'{key}_layer_{layer_idx}_latent'

    if latent_key not in batch_data:
        # Try alternative format
        latent_key = f'{key}_layer_{layer_idx}'
        if latent_key not in batch_data:
            print(f"Warning: No latent data found for {key} layer {layer_idx}. "
                  f"Run evaluation with --save-latents and multiple --layer-indices.")
            return None

    latent = batch_data[latent_key][sample_idx]  # [C, H, W]
    num_channels = latent.shape[0]

    # Calculate grid dimensions
    nrows = ceil(num_channels / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))

    if nrows == 1:
        ax_flat = [axs] if ncols == 1 else list(axs)
    else:
        ax_flat = axs.flatten()

    # Turn off all axes first
    for ax in ax_flat:
        ax.axis('off')

    for j in range(num_channels):
        if j >= len(ax_flat):
            break

        ax = ax_flat[j]

        # Get min/max for this channel for better visualization
        vmin = latent[j].min()
        vmax = latent[j].max()

        im = ax.imshow(latent[j], cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f'Ch {j}', fontsize=8)
        ax.axis('off')

    # Add overall title
    flip_str = " + flip" if flip else ""
    fig.suptitle(f'Latent Features Layer {layer_idx} (rotation={rotation}°{flip_str})',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Latent grid saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_latent_comparison_grid(
    batch_data: Dict,
    sample_idx: int = 0,
    rotation: int = 0,
    flip: bool = False,
    layer_idx: int = -1,
    ncols: int = 6,
    save_path: Optional[str] = None,
):
    """
    Plot comparison of original vs transformed latents side by side with difference.

    Shows original latent, transformed (rotated back) latent, and their difference
    for each channel.

    Args:
        batch_data: Dictionary with batch results
        sample_idx: Sample index within batch
        rotation: Rotation angle
        flip: Whether flip was applied
        layer_idx: Layer index for latent
        ncols: Number of columns (should be multiple of 3)
        save_path: Path to save figure
    """
    # Get original latent (r0_nf)
    orig_key = f'r0_nf_layer_{layer_idx}_latent'
    if orig_key not in batch_data:
        orig_key = f'r0_nf_layer_{layer_idx}'
        if orig_key not in batch_data:
            print(f"Warning: No original latent data found for layer {layer_idx}.")
            return None

    # Get transformed latent
    trans_key_base = f"r{rotation}" + ("_f" if flip else "_nf")
    trans_key = f'{trans_key_base}_layer_{layer_idx}_latent'
    if trans_key not in batch_data:
        trans_key = f'{trans_key_base}_layer_{layer_idx}'
        if trans_key not in batch_data:
            print(f"Warning: No transformed latent data found for {trans_key_base} layer {layer_idx}.")
            return None

    latent_orig = batch_data[orig_key][sample_idx]  # [C, H, W]
    latent_trans = batch_data[trans_key][sample_idx]  # [C, H, W]

    num_channels = latent_orig.shape[0]

    # Calculate grid dimensions (3 columns per channel: orig, trans, diff)
    nrows = ceil(num_channels / (ncols // 3))
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))

    if nrows == 1:
        ax_flat = list(axs) if ncols > 1 else [axs]
    else:
        ax_flat = axs.flatten()

    # Turn off all axes first
    for ax in ax_flat:
        ax.axis('off')

    for i in range(0, len(ax_flat), 3):
        j = i // 3

        if j >= num_channels:
            break

        ax_orig = ax_flat[i]
        ax_trans = ax_flat[i + 1]
        ax_diff = ax_flat[i + 2]

        # Get consistent scale for orig and trans
        vmin = min(latent_orig[j].min(), latent_trans[j].min())
        vmax = max(latent_orig[j].max(), latent_trans[j].max())

        # Original latent
        ax_orig.imshow(latent_orig[j], cmap='viridis', vmin=vmin, vmax=vmax)
        ax_orig.set_title(f'Original\nCh {j}', fontsize=8)
        ax_orig.axis('off')

        # Transformed latent (rotated back)
        ax_trans.imshow(latent_trans[j], cmap='viridis', vmin=vmin, vmax=vmax)
        ax_trans.set_title(f'Transformed\nCh {j}', fontsize=8)
        ax_trans.axis('off')

        # Difference
        diff = np.abs(latent_orig[j] - latent_trans[j])
        ax_diff.imshow(diff, cmap='hot')
        ax_diff.set_title(f'|Diff|\nCh {j}', fontsize=8)
        ax_diff.axis('off')

    # Add overall title
    flip_str = " + flip" if flip else ""
    fig.suptitle(f'Latent Comparison Layer {layer_idx} (rotation={rotation}°{flip_str})',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Latent comparison grid saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize equivariance evaluation results')
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Directory containing evaluation results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save visualizations (default: results_dir/plots)'
    )
    parser.add_argument(
        '--batch-idx',
        type=int,
        default=0,
        help='Batch index for detailed visualizations'
    )
    parser.add_argument(
        '--sample-idx',
        type=int,
        default=0,
        help='Sample index within batch'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1,
        help='Number of samples to visualize (creates plots for multiple samples)'
    )
    parser.add_argument(
        '--rotations',
        type=int,
        nargs='+',
        default=[30, 45, 90, 135, 180, 270],
        help='Rotation angles to visualize (e.g., --rotations 90 180 270)'
    )
    parser.add_argument(
        '--channel-idx',
        type=int,
        default=0,
        help='Channel index to visualize'
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.results_dir}")
    aggregated_metrics, batch_results = load_results(args.results_dir)

    # Create output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.results_dir) / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {output_dir}")

    # Plot metrics comparison
    print("\nGenerating metrics comparison...")
    plot_metrics_comparison(
        aggregated_metrics,
        save_path=str(output_dir / 'metrics_comparison.png')
    )

    # Check if batch data has necessary keys
    if batch_results and len(batch_results) > args.batch_idx:
        batch_data = batch_results[args.batch_idx]

        # Determine how many samples to process
        num_samples_in_batch = len(batch_data.get('img_paths', []))
        if num_samples_in_batch == 0:
            # Try to infer from data shape
            for key in batch_data.keys():
                if '_img' in key and batch_data[key].ndim >= 4:
                    num_samples_in_batch = batch_data[key].shape[0]
                    break

        samples_to_process = min(args.num_samples, num_samples_in_batch)
        print(f"\nProcessing {samples_to_process} sample(s) from batch {args.batch_idx}")

        # Plot latent and reconstruction comparisons for multiple samples
        for sample_idx in range(args.sample_idx, args.sample_idx + samples_to_process):
            if sample_idx >= num_samples_in_batch:
                print(f"Warning: Sample index {sample_idx} exceeds batch size, stopping.")
                break

            print(f"\n{'='*80}")
            print(f"Processing sample {sample_idx} (in batch {args.batch_idx})")
            print(f"{'='*80}")

            # Plot original (no rotation) reconstruction grid first
            if 'r0_nf_img' in batch_data:
                print(f"\nGenerating original reconstruction grid (no rotation)...")
                plot_reconstruction_grid(
                    batch_data,
                    sample_idx=sample_idx,
                    rotation=0,
                    flip=False,
                    save_path=str(output_dir / f'reconstruction_grid_sample{sample_idx}_r0.png')
                )

            # Plot original latent grid (no rotation)
            print(f"\nGenerating original latent grid (no rotation)...")
            plot_latent_grid(
                batch_data,
                sample_idx=sample_idx,
                rotation=0,
                flip=False,
                layer_idx=-1,
                save_path=str(output_dir / f'latent_grid_sample{sample_idx}_r0.png')
            )

            for rotation in args.rotations:
                print(f"\nGenerating latent comparison for {rotation}° rotation...")
                plot_latent_comparison(
                    batch_data,
                    batch_idx=sample_idx,
                    rotation=rotation,
                    channel_idx=args.channel_idx,
                    save_path=str(output_dir / f'latent_comparison_sample{sample_idx}_r{rotation}_ch{args.channel_idx}.png')
                )

                # Plot reconstruction comparison
                if f'r{rotation}_nf_img' in batch_data:
                    print(f"Generating reconstruction comparison for {rotation}° rotation...")
                    plot_reconstruction_comparison(
                        batch_data,
                        batch_idx=sample_idx,
                        rotation=rotation,
                        channel_idx=args.channel_idx,
                        save_path=str(output_dir / f'reconstruction_comparison_sample{sample_idx}_r{rotation}_ch{args.channel_idx}.png')
                    )

                    # Plot full reconstruction grid with all channels
                    print(f"Generating reconstruction grid for {rotation}° rotation...")
                    plot_reconstruction_grid(
                        batch_data,
                        sample_idx=sample_idx,
                        rotation=rotation,
                        flip=False,
                        save_path=str(output_dir / f'reconstruction_grid_sample{sample_idx}_r{rotation}.png')
                    )

                # Plot latent grid for this rotation
                print(f"Generating latent grid for {rotation}° rotation...")
                plot_latent_grid(
                    batch_data,
                    sample_idx=sample_idx,
                    rotation=rotation,
                    flip=False,
                    layer_idx=-1,
                    save_path=str(output_dir / f'latent_grid_sample{sample_idx}_r{rotation}.png')
                )

                # Plot latent comparison grid (original vs transformed)
                print(f"Generating latent comparison grid for {rotation}° rotation...")
                plot_latent_comparison_grid(
                    batch_data,
                    sample_idx=sample_idx,
                    rotation=rotation,
                    flip=False,
                    layer_idx=-1,
                    save_path=str(output_dir / f'latent_comparison_grid_sample{sample_idx}_r{rotation}.png')
                )

        # Plot heatmap (uses all batches)
        print("\nGenerating equivariance error heatmap...")
        plot_equivariance_heatmap(
            batch_results,
            metric='equiv_mse',
            save_path=str(output_dir / 'equivariance_heatmap.png')
        )

    print("\n" + "="*80)
    print("Visualization complete!")
    print(f"All plots saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
