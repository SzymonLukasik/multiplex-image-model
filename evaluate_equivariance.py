"""
Rotation Equivariance Evaluation Script

Evaluates the model's rotation equivariance by:
1. Testing various rotations (90°, 180°, 270°, 30°, 45°, 135°)
2. Computing equivariance errors between latent representations
3. Calculating metrics including MSE and cosine similarity
4. Extracting features from all layers for analysis
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

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


def circular_mask(H: int, W: int, device: torch.device) -> torch.Tensor:
    """
    Create largest inscribed circle mask for an H×W feature map.

    Args:
        H: Height of the feature map
        W: Width of the feature map
        device: Device to create the mask on

    Returns:
        Binary mask with circle inscribed [H, W]
    """
    Y, X = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    r2 = (X - W // 2) ** 2 + (Y - H // 2) ** 2
    return (r2 <= (min(H, W) // 2) ** 2).float()


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
        # Use exact rotation for multiples of 90°
        k = int(round(theta_deg / 90)) % 4
        return torch.rot90(x, k=k, dims=(-2, -1))

    # Use torchvision for non-90° rotations
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
    # Flatten spatial dimensions
    flat1 = latent1.flatten(1)  # [B, C*H*W]
    flat2 = latent2.flatten(1)  # [B, C*H*W]

    # Compute cosine similarity
    cosine_sim = F.cosine_similarity(flat1, flat2, dim=1)  # [B]
    return cosine_sim


def apply_transformation(
    img: torch.Tensor,
    rotation: float,
    flip: bool,
    use_mask: bool,
    device: torch.device
) -> torch.Tensor:
    """
    Apply rotation, flip, and optional circular masking to an image.

    Args:
        img: Input image [B, C, H, W]
        rotation: Rotation angle in degrees
        flip: Whether to apply horizontal flip
        use_mask: Whether to apply circular mask
        device: Device for computation

    Returns:
        Transformed image [B, C, H, W]
    """
    # Apply circular mask if needed (before rotation for non-90° angles)
    if use_mask and rotation % 90 != 0:
        mask = circular_mask(img.shape[2], img.shape[3], device=device)
        img = img * mask.unsqueeze(0).unsqueeze(0)

    # Apply rotation
    if rotation > 0 and rotation % 90 == 0:
        img = torch.rot90(img, k=int(rotation // 90), dims=[-2, -1])
    elif rotation > 0:
        img = rotate_tensor(img, rotation, mode='bilinear')

    # Apply flip
    if flip:
        img = torch.flip(img, dims=[3])

    # Apply circular mask if needed (after rotation for 90° angles)
    if use_mask and rotation % 90 == 0:
        mask = circular_mask(img.shape[2], img.shape[3], device=device)
        img = img * mask.unsqueeze(0).unsqueeze(0)

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
    # Inverse flip
    if flip:
        latent = torch.flip(latent, dims=[3])

    # Inverse rotation
    if rotation > 0 and rotation % 90 == 0:
        latent = torch.rot90(latent, k=int(-rotation // 90), dims=[-2, -1])
    elif rotation > 0:
        latent = rotate_tensor(latent, -rotation, mode='bilinear')

    return latent

def to_tensor(x):
    """Convert GeometricTensor or regular tensor to regular PyTorch tensor."""
    if hasattr(x, 'tensor'):
        # It's a GeometricTensor from escnn
        return x.tensor
    return x

def evaluate_single_batch(
    model: torch.nn.Module,
    org_image: torch.Tensor,
    channel_ids: torch.Tensor,
    device: torch.device,
    rotations: List[Tuple[float, bool]] = None,
    layer_indices: List[int] = None,
    save_latents: bool = False
) -> Dict:
    """
    Evaluate equivariance for a single batch with various rotations.

    Args:
        model: The trained model
        org_image: Original images [B, C, H, W]
        channel_ids: Channel IDs [B, C]
        device: Device for computation
        rotations: List of (rotation_angle, flip) tuples to test
        layer_indices: List of layer indices to evaluate (default: [-1] for last layer only)
                      Use negative indices for counting from the end

    Returns:
        Dictionary with results for each transformation
    """
    if layer_indices is None:
        layer_indices = [-1]  # Default: only evaluate the last (trivial) layer
    if rotations is None:
        rotations = [
            (0, False),      # Original
            (90, False),     # 90° rotation
            (180, False),    # 180° rotation
            (270, False),    # 270° rotation
            (30, False),     # 30° rotation
            (45, False),     # 45° rotation
            (135, False),    # 135° rotation
            (0, True),       # Horizontal flip
            (90, True),      # 90° + flip
        ]

    results = {}
    org_image = org_image.to(torch.float32).to(device)
    batch_size = org_image.shape[0]
    num_channels = org_image.shape[1]

    # Storage for reference latents (one per layer)
    latent_org_dict = {}  # {layer_idx: tensor}
    latent_mask_dict = {}  # {layer_idx: tensor}

    compute_all_layers = len(layer_indices) > 1 or save_latents

    # First pass: compute original (no rotation, no flip) and masked reference
    for use_mask in [True, False]:
        img = org_image
        if use_mask:
            mask = circular_mask(org_image.shape[2], org_image.shape[3], device=device)
            img = org_image * mask.unsqueeze(0).unsqueeze(0)

        key = "r0_nf" + ("_mask" if use_mask else "")
        channel_ids_batch = torch.arange(num_channels).unsqueeze(0).expand(batch_size, -1).to(device)

        # Use autocast only if CUDA is available and autocast is supported
        autocast_context = cuda_autocast() if (HAS_AUTOCAST and 'cuda' in str(device)) else nullcontext()

        with torch.no_grad():
            with autocast_context:
                output = model(img, channel_ids_batch, channel_ids_batch, True)
                features = output["features"]
                reconstructions = output["output"]

                # Crop reconstruction to match input
                reconstructions = reconstructions[:, :, 3:-4, 3:-4]
                mi, logsigma = reconstructions.unbind(dim=-1)
                mi = torch.sigmoid(mi)

        # Store reference latents for requested layers
        for layer_idx in layer_indices:
            layer_tensor = to_tensor(features[layer_idx]).cpu().detach().float()
            if use_mask:
                latent_mask_dict[layer_idx] = layer_tensor
            else:
                latent_org_dict[layer_idx] = layer_tensor

        # Store basic outputs (only once, not per layer)
        if key not in results:
            results[key] = {
                'reconstruction': mi.cpu().detach().float().numpy(),
                'logsigma': logsigma.cpu().detach().float().numpy(),
                'img': img.cpu().detach().numpy(),
            }

    # Second pass: compute all transformations and equivariance errors
    for rotation, flip in rotations:
        # Skip the original case (already computed)
        if rotation == 0 and not flip:
            continue

        # Determine if we need mask for this rotation
        use_mask = (rotation % 90 != 0)

        # Apply transformation
        img = apply_transformation(org_image, rotation, flip, use_mask, device)

        key = f"r{int(rotation)}" + ("_f" if flip else "_nf")
        channel_ids_batch = torch.arange(num_channels).unsqueeze(0).expand(batch_size, -1).to(device)

        # Use autocast only if CUDA is available and autocast is supported
        autocast_context = cuda_autocast() if (HAS_AUTOCAST and 'cuda' in str(device)) else nullcontext()

        with torch.no_grad():
            with autocast_context:
                output = model(img, channel_ids_batch, channel_ids_batch, True)
                features = output["features"]
                reconstructions = output["output"]

                reconstructions = reconstructions[:, :, 3:-4, 3:-4]
                mi, logsigma = reconstructions.unbind(dim=-1)
                mi = torch.sigmoid(mi)

        # Compute metrics for each requested layer
        layer_metrics = {}

        for layer_idx in layer_indices:
            # Get latent and apply inverse transformation
            latent = to_tensor(features[layer_idx]).cpu().detach().float()
            latent_inv = inverse_transformation(latent, rotation, flip)

            # Choose reference based on rotation type
            reference = latent_mask_dict[layer_idx] if rotation % 90 != 0 else latent_org_dict[layer_idx]

            # MSE equivariance error
            equiv_mse = ((latent_inv - reference) ** 2).mean(dim=[1, 2, 3])  # [B]

            # Cosine similarity
            cosine_sim = compute_cosine_similarity(latent_inv, reference)  # [B]

            # L1 error
            equiv_l1 = (latent_inv - reference).abs().mean(dim=[1, 2, 3])  # [B]

            # Store metrics for this layer
            layer_key = f'layer_{layer_idx}' if compute_all_layers else ''
            prefix = f'{layer_key}_' if layer_key else ''

            layer_metrics[f'{prefix}equiv_mse'] = equiv_mse.numpy()
            layer_metrics[f'{prefix}equiv_l1'] = equiv_l1.numpy()
            layer_metrics[f'{prefix}cosine_sim'] = cosine_sim.numpy()

            # Store latent if requested
            if compute_all_layers:
                layer_metrics[f'{layer_key}_latent'] = latent_inv.numpy()

        # Store results
        results[key] = {
            'reconstruction': mi.cpu().detach().float().numpy(),
            'logsigma': logsigma.cpu().detach().float().numpy(),
            'img': img.cpu().detach().numpy(),
            **layer_metrics
        }

    # Store original latents for visualization
    if compute_all_layers:
        for layer_idx in layer_indices:
            results['r0_nf'][f'layer_{layer_idx}_latent'] = latent_org_dict[layer_idx].numpy()

    return results


def aggregate_results(all_results: List[Dict]) -> Dict:
    """
    Aggregate results across all batches.

    Args:
        all_results: List of result dictionaries from each batch

    Returns:
        Dictionary with aggregated metrics
    """
    aggregated = {}

    # Collect all keys
    all_keys = set()
    for results in all_results:
        all_keys.update(results.keys())

    # Aggregate metrics for each transformation
    for key in all_keys:
        if key in ['r0_nf', 'r0_nf_mask']:
            continue  # Skip reference cases

        # Collect metrics
        equiv_mse_list = []
        equiv_l1_list = []
        cosine_sim_list = []

        for results in all_results:
            if key in results and 'equiv_mse' in results[key]:
                equiv_mse_list.append(results[key]['equiv_mse'])
                equiv_l1_list.append(results[key]['equiv_l1'])
                cosine_sim_list.append(results[key]['cosine_sim'])

        if equiv_mse_list:
            # Concatenate and compute statistics
            equiv_mse = np.concatenate(equiv_mse_list)
            equiv_l1 = np.concatenate(equiv_l1_list)
            cosine_sim = np.concatenate(cosine_sim_list)

            aggregated[key] = {
                'equiv_mse_mean': float(equiv_mse.mean()),
                'equiv_mse_std': float(equiv_mse.std()),
                'equiv_mse_median': float(np.median(equiv_mse)),
                'equiv_l1_mean': float(equiv_l1.mean()),
                'equiv_l1_std': float(equiv_l1.std()),
                'cosine_sim_mean': float(cosine_sim.mean()),
                'cosine_sim_std': float(cosine_sim.std()),
                'cosine_sim_median': float(np.median(cosine_sim)),
            }

    return aggregated


def main():
    parser = argparse.ArgumentParser(description='Evaluate rotation equivariance')
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
        help='Layer indices to evaluate (default: -1 for last layer only). Use negative indices to count from end.'
    )
    parser.add_argument(
        '--save-latents',
        action='store_true',
        help='Save latent representations for visualization (even with single layer evaluation)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from config: {args.config}")
    print(f"Using device: {args.device}")

    # Load model and data
    model, train_dataloader, test_dataloader, TOKENIZER, INV_TOKENIZER, config = load_model_and_data(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    print("\n" + "="*80)
    print("Starting Rotation Equivariance Evaluation")
    print("="*80)


    # Evaluation loop
    all_results = []
    num_batches = args.num_batches or len(test_dataloader)

    with torch.no_grad():
        for i, (org_image, channel_ids, panel_idx, img_path) in enumerate(tqdm(
            test_dataloader,
            total=num_batches,
            desc="Evaluating"
        )):
            if i >= num_batches:
                break

            results = evaluate_single_batch(
                model=model,
                org_image=org_image,
                channel_ids=channel_ids,
                device=args.device if args.device != 'cpu' else model.parameters().__next__().device,
                layer_indices=args.layer_indices,
                save_latents=args.save_latents
            )

            # Add metadata
            results['img_paths'] = img_path
            # Handle panel_idx - it might be tensor, tuple, or other type
            if isinstance(panel_idx, torch.Tensor):
                results['panel_idx'] = panel_idx.numpy()
            else:
                results['panel_idx'] = panel_idx  # Already a tuple or list

            all_results.append(results)

    print("\n" + "="*80)
    print("Computing Aggregate Metrics")
    print("="*80)

    # Aggregate results
    aggregated_metrics = aggregate_results(all_results)

    # Print results
    print("\nEquivariance Metrics:")
    print("-" * 80)
    for key, metrics in sorted(aggregated_metrics.items()):
        print(f"\n{key}:")
        print(f"  MSE:            {metrics['equiv_mse_mean']:.6f} ± {metrics['equiv_mse_std']:.6f}")
        print(f"  MSE (median):   {metrics['equiv_mse_median']:.6f}")
        print(f"  L1:             {metrics['equiv_l1_mean']:.6f} ± {metrics['equiv_l1_std']:.6f}")
        print(f"  Cosine Sim:     {metrics['cosine_sim_mean']:.6f} ± {metrics['cosine_sim_std']:.6f}")
        print(f"  Cosine (median): {metrics['cosine_sim_median']:.6f}")

    # Save aggregated metrics
    metrics_path = output_dir / 'aggregated_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(aggregated_metrics, f, indent=2)
    print(f"\nAggregated metrics saved to: {metrics_path}")

    # Save full results if requested
    if args.save_reconstructions or args.save_features or args.save_latents:
        print("\nSaving detailed results...")

        # Create separate file for each batch to avoid memory issues
        for i, results in enumerate(tqdm(all_results, desc="Saving")):
            batch_file = output_dir / f'batch_{i:04d}.npz'

            # Prepare data to save
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

                # Save latents if requested
                if args.save_latents:
                    for sub_key in value:
                        if '_latent' in sub_key:
                            save_dict[f'{key}_{sub_key}'] = value[sub_key]

                # Always save metrics
                for metric_key in ['equiv_mse', 'equiv_l1', 'cosine_sim']:
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
