#!/usr/bin/env python
"""
Create latent visualization plots from equivariance evaluation results.

Usage:
    python scripts/plot_latents.py --results-dir <results_dir> [options]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path


def plot_latent_grid(
    batch_data,
    sample_idx=0,
    rotations=[90, 180, 270, 45],
    channel_indices=[0, 1, 2],
    layer_idx=-1,
    save_path=None,
    figsize=(20, 12)
):
    """
    Create a grid of latent visualizations across rotations and channels.

    Args:
        batch_data: Dictionary with batch results
        sample_idx: Sample index within batch
        rotations: List of rotation angles to show
        channel_indices: List of latent channel indices to show
        layer_idx: Layer index
        save_path: Path to save figure
        figsize: Figure size
    """
    n_rotations = len(rotations)
    n_channels = len(channel_indices)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_channels + 1, n_rotations + 1, figure=fig, hspace=0.3, wspace=0.3)

    # Get reference key
    key_reference = "r0_nf"
    layer_key = f'layer_{layer_idx}_latent'

    # Check if latents exist
    if f'{key_reference}_{layer_key}' not in batch_data:
        print(f"Error: Latent features not found. Make sure to run evaluation with --save-latents")
        return

    ref_latent = batch_data[f'{key_reference}_{layer_key}'][sample_idx]

    # Title row
    fig.text(0.15, 0.95, 'Original', ha='center', fontsize=12, fontweight='bold')
    for col_idx, rotation in enumerate(rotations):
        fig.text(0.15 + (col_idx + 1) * (0.7 / (n_rotations + 1)), 0.95,
                f'{rotation}°', ha='center', fontsize=12, fontweight='bold')

    # Plot each channel
    for row_idx, channel_idx in enumerate(channel_indices):
        # Label row
        ax_label = fig.add_subplot(gs[row_idx, 0])
        ax_label.text(0.5, 0.5, f'Channel {channel_idx}',
                     ha='center', va='center', fontsize=11, fontweight='bold', rotation=90)
        ax_label.axis('off')

        # Original latent
        ax_orig = fig.add_subplot(gs[row_idx, 1])
        im = ax_orig.imshow(ref_latent[channel_idx], cmap='viridis')
        ax_orig.axis('off')
        if row_idx == 0:
            ax_orig.set_title('Original', fontsize=10)
        plt.colorbar(im, ax=ax_orig, fraction=0.046, pad=0.04)

        # Rotated latents
        for col_idx, rotation in enumerate(rotations):
            use_mask = rotation % 90 != 0
            key_transformed = f"r{rotation}_nf"
            key_ref = "r0_nf" + ("_mask" if use_mask else "")

            if f'{key_transformed}_{layer_key}' not in batch_data:
                continue

            trans_latent = batch_data[f'{key_transformed}_{layer_key}'][sample_idx]
            ref_for_comp = batch_data[f'{key_ref}_{layer_key}'][sample_idx] if use_mask else ref_latent

            ax = fig.add_subplot(gs[row_idx, col_idx + 2])
            im = ax.imshow(trans_latent[channel_idx], cmap='viridis')
            ax.axis('off')
            if row_idx == 0:
                ax.set_title(f'{rotation}°', fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Add difference row
    fig.text(0.05, 0.05 + (0.9 / (n_channels + 1)) / 2, 'Difference\n(abs)',
             ha='center', va='center', fontsize=11, fontweight='bold', rotation=90)

    for col_idx, rotation in enumerate(rotations):
        use_mask = rotation % 90 != 0
        key_transformed = f"r{rotation}_nf"
        key_ref = "r0_nf" + ("_mask" if use_mask else "")

        if f'{key_transformed}_{layer_key}' not in batch_data:
            continue

        trans_latent = batch_data[f'{key_transformed}_{layer_key}'][sample_idx]
        ref_for_comp = batch_data[f'{key_ref}_{layer_key}'][sample_idx] if use_mask else ref_latent

        # Average difference across channels
        diff = np.abs(trans_latent[channel_indices] - ref_for_comp[channel_indices]).mean(axis=0)

        ax = fig.add_subplot(gs[n_channels, col_idx + 2])
        im = ax.imshow(diff, cmap='hot')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add MSE text
        mse = ((trans_latent - ref_for_comp) ** 2).mean()
        ax.text(0.5, -0.1, f'MSE: {mse:.4f}', ha='center', transform=ax.transAxes, fontsize=8)

    plt.suptitle(f'Latent Representations Across Rotations (Sample {sample_idx}, Layer {layer_idx})',
                 fontsize=14, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved latent grid to: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Create latent visualization plots')
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Directory with evaluation results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory for plots (default: results_dir/latent_plots)'
    )
    parser.add_argument(
        '--batch-idx',
        type=int,
        default=0,
        help='Batch index (default: 0)'
    )
    parser.add_argument(
        '--sample-idx',
        type=int,
        default=0,
        help='Sample index within batch (default: 0)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1,
        help='Number of samples to plot (default: 1)'
    )
    parser.add_argument(
        '--rotations',
        type=int,
        nargs='+',
        default=[90, 180, 270, 45],
        help='Rotation angles to visualize'
    )
    parser.add_argument(
        '--channels',
        type=int,
        nargs='+',
        default=[0, 1, 2],
        help='Latent channel indices to visualize'
    )
    parser.add_argument(
        '--layer-idx',
        type=int,
        default=-1,
        help='Layer index (default: -1 for last layer)'
    )

    args = parser.parse_args()

    # Load results
    results_path = Path(args.results_dir)
    batch_file = results_path / f'batch_{args.batch_idx:04d}.npz'

    if not batch_file.exists():
        print(f"Error: Batch file not found: {batch_file}")
        print(f"Make sure to run evaluation with --save-latents flag")
        return

    print(f"Loading batch data from: {batch_file}")
    batch_data = dict(np.load(batch_file, allow_pickle=True))

    # Create output directory
    output_dir = Path(args.output_dir) if args.output_dir else results_path / 'latent_plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {output_dir}")

    # Determine number of samples in batch
    layer_key = f'layer_{args.layer_idx}_latent'
    ref_key = f'r0_nf_{layer_key}'

    if ref_key not in batch_data:
        print(f"Error: Latent features not found in batch data.")
        print(f"Expected key: {ref_key}")
        print(f"Available keys: {list(batch_data.keys())[:10]}...")
        print(f"\nMake sure to run evaluation with --save-latents flag:")
        print(f"  python evaluate_equivariance.py --save-latents ...")
        return

    num_samples_in_batch = batch_data[ref_key].shape[0]
    samples_to_process = min(args.num_samples, num_samples_in_batch - args.sample_idx)

    print(f"\nProcessing {samples_to_process} sample(s) starting from index {args.sample_idx}")
    print(f"Rotations: {args.rotations}")
    print(f"Channels: {args.channels}")

    # Plot each sample
    for sample_idx in range(args.sample_idx, args.sample_idx + samples_to_process):
        print(f"\nPlotting sample {sample_idx}...")

        save_path = output_dir / f'latent_grid_sample{sample_idx}_layer{args.layer_idx}.png'

        plot_latent_grid(
            batch_data=batch_data,
            sample_idx=sample_idx,
            rotations=args.rotations,
            channel_indices=args.channels,
            layer_idx=args.layer_idx,
            save_path=str(save_path)
        )

    print(f"\n{'='*80}")
    print(f"Latent plotting complete!")
    print(f"Plots saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
