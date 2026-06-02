"""
Load trained model for evaluation - aligned with train_masked_model.py
"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomRotation, RandomCrop, RandomHorizontalFlip
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler, TestCrop
from multiplex_model.losses import nll_loss
from multiplex_model.utils import ClampWithGrad, plot_reconstructs_with_masks
from multiplex_model.modules import MultiplexAutoencoder


def load_model_and_data(config_path, checkpoint_path=None, device='cpu'):
    """
    Load model and data for evaluation, aligned with training script.

    Args:
        config_path: Path to the YAML configuration file
        checkpoint_path: Path to model checkpoint (optional, overrides config)
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        model, train_dataloader, test_dataloader, TOKENIZER, INV_TOKENIZER, config
    """
    yaml = YAML(typ='safe')
    with open(config_path, 'r') as f:
        config = yaml.load(f)

    # Override device if specified
    config['device'] = device

    print(f'Using device: {device}')

    SIZE = config['input_image_size']
    print(f"INPUT IMAGE SIZE: {SIZE}")
    BATCH_SIZE = config.get('batch_size', 4)
    NUM_WORKERS = config.get('num_workers', 4)

    # Load panel configuration and tokenizer
    PANEL_CONFIG = YAML().load(open(config['panel_config']))
    TOKENIZER_CONFIG = YAML().load(open(config['tokenizer_config']))
    print(f"Training on datasets: {PANEL_CONFIG['datasets']}")

    MARKERS_SET = {k for dataset in PANEL_CONFIG['datasets'] for k in PANEL_CONFIG['markers'][dataset]}
    print(f"Markers set: {MARKERS_SET}")
    print(f"Number of markers: {len(MARKERS_SET)}")

    # Use the tokenizer from the config file (must match training!)
    # NOTE: Python's sorted() uses case-sensitive ASCII order, which differs from
    # the case-insensitive order used when the tokenizer was originally built.
    # Rebuilding from sorted(MARKERS_SET) scrambles 261/265 marker IDs.
    TOKENIZER = TOKENIZER_CONFIG
    INV_TOKENIZER = {v: k for k, v in TOKENIZER.items()}

    # Data transforms - aligned with training script
    train_transform = Compose([
        RandomRotation(180, interpolation=InterpolationMode.BILINEAR),
        RandomCrop(SIZE),
        RandomHorizontalFlip(),
    ])

    test_transform = TestCrop(SIZE[0])

    # Create datasets - aligned with training script parameters
    train_dataset = DatasetFromTIFF(
        panels_config=PANEL_CONFIG,
        split='train',
        marker_tokenizer=TOKENIZER,
        transform=train_transform,
        use_preprocessing=False,  # saved data is already preprocessed
        use_median_denoising=False,
        use_butterworth_filter=True,
        use_minmax_normalization=False,
        use_clip_normalization=True,
        file_extension='npy'
    )

    test_dataset = DatasetFromTIFF(
        panels_config=PANEL_CONFIG,
        split='test',
        marker_tokenizer=TOKENIZER,
        transform=test_transform,
        use_preprocessing=False,  # saved data is already preprocessed
        use_median_denoising=False,
        use_butterworth_filter=True,
        use_minmax_normalization=False,
        use_clip_normalization=True,
        file_extension='npy'
    )

    # Create batch samplers and dataloaders
    train_batch_sampler = PanelBatchSampler(train_dataset, BATCH_SIZE)
    test_batch_sampler = PanelBatchSampler(test_dataset, BATCH_SIZE, shuffle=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True if device != 'cpu' else False,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=4 if NUM_WORKERS > 0 else None
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_sampler=test_batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True if device != 'cpu' else False,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=4 if NUM_WORKERS > 0 else None
    )

    # Model configuration - aligned with training script (no superkernel_config)
    model_config = {
        'num_channels': len(TOKENIZER),
        'encoder_config': config['encoder'],
        'decoder_config': config['decoder'],
    }

    # Create model based on type
    if config["model_type"] == "FullyEquivariantConvnext":
        from train_masked_model_ddp_ed import FullyEquivariantMultiplexAutoencoder
        model = FullyEquivariantMultiplexAutoencoder(**model_config).to(device)
    elif config["model_type"] == "EquivariantConvnext":
        from multiplex_model.equivariant_modules import EquivariantMultiplexAutoencoder
        model = EquivariantMultiplexAutoencoder(**model_config).to(device)
    elif config["model_type"] == "EquivariantConvnextV2":
        from multiplex_model.equivariant_modules_v2 import EquivariantMultiplexAutoencoder
        model = EquivariantMultiplexAutoencoder(**model_config).to(device)
    elif config["model_type"] == "Convnext":
        model = MultiplexAutoencoder(**model_config).to(device)
    elif config["model_type"] == "ConvnextImmuVisLegacy":
        # Original dav3794/multiplex-image-model master ImmuVis architecture
        # (encoder norm + latent_norm, Identity pm-stem, scaling 2^(ma+pm-1)).
        from multiplex_model.modules_immuvis_legacy import MultiplexAutoencoderLegacy
        model = MultiplexAutoencoderLegacy(**model_config).to(device)
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")

    print(f'Model created with config: {model_config}')
    
    print(f"Model: {model}")
    print(f'Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters')
    print(f'Training on {len(train_dataloader.dataset)} training samples and {len(test_dataloader.dataset)} test samples')
    print(f'Batch size: {BATCH_SIZE}, Number of workers: {NUM_WORKERS}')

    # Load checkpoint if provided
    checkpoint_to_load = checkpoint_path or config.get('from_checkpoint')
    if checkpoint_to_load:
        print(f'Loading model from checkpoint: {checkpoint_to_load}')
        checkpoint = torch.load(checkpoint_to_load, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded checkpoint from epoch {checkpoint.get("epoch", "unknown")}')

    model.eval()  # Set to evaluation mode

    return model, train_dataloader, test_dataloader, TOKENIZER, INV_TOKENIZER, config


def evaluate_batch(model, img, channel_ids, device,
                   spatial_masking_ratio=0.6,
                   fully_masked_channels_max_frac=0.5,
                   mask_patch_size=8):
    """
    Evaluate a single batch with masking (aligned with training script).

    Args:
        model: The trained model
        img: Input images [B, C, H, W]
        channel_ids: Channel IDs [B, C]
        device: Device to run on
        spatial_masking_ratio: Fraction of patches to mask spatially
        fully_masked_channels_max_frac: Max fraction of channels to fully mask
        mask_patch_size: Size of patches for spatial masking

    Returns:
        Dictionary with outputs and metrics
    """
    batch_size, num_channels, H, W = img.shape
    img = img.to(device, dtype=torch.float32)
    channel_ids = channel_ids.to(device, dtype=torch.long)

    # Sample full channels to mask (drop)
    max_channels_to_mask = int(np.ceil(num_channels * fully_masked_channels_max_frac))
    num_channels_to_mask = np.random.randint(1, max_channels_to_mask + 1)

    masked_img = []
    active_channel_ids = []
    for b_i in range(batch_size):
        channels_to_keep = torch.randperm(num_channels)[num_channels_to_mask:]
        masked_img.append(img[b_i:b_i+1, channels_to_keep, :, :])
        active_channel_ids.append(channel_ids[b_i:b_i+1, channels_to_keep])

    masked_img = torch.cat(masked_img, dim=0)  # [B, C_new, H, W]
    active_channel_ids = torch.cat(active_channel_ids, dim=0)  # [B, C_new]
    num_active_channels = masked_img.shape[1]

    # Randomly mask spatial_masking_ratio image patches
    h = w = H // mask_patch_size
    mask = torch.rand((batch_size, num_active_channels, h, w), device=masked_img.device)
    mask = mask <= spatial_masking_ratio
    pixel_mask = mask.repeat_interleave(mask_patch_size, dim=2).repeat_interleave(mask_patch_size, dim=3)

    masked_img[pixel_mask] = 0.0  # mask patches by setting to zero

    with torch.no_grad():
        output = model(masked_img, active_channel_ids, channel_ids)['output']
        mi, logsigma = output.unbind(dim=-1)
        mi = torch.sigmoid(mi)

        logsigma = ClampWithGrad.apply(logsigma, -15.0, 15.0)
        loss = nll_loss(img, mi, logsigma)
        mae = torch.abs(img - mi).mean()
        mse = torch.square(img - mi).mean()

    # Get unmasked channels for plotting
    unactive_channels = [i for i in channel_ids[0] if i not in active_channel_ids[0]]

    return {
        'loss': loss.item(),
        'mae': mae.item(),
        'mse': mse.item(),
        'reconstruction': mi,
        'logsigma': logsigma,
        'pixel_mask': pixel_mask,
        'active_channel_ids': active_channel_ids,
        'unactive_channels': unactive_channels,
        'original': img,
    }


# Example usage for Jupyter notebook
if __name__ == '__main__':
    # Configuration
    config_path = "/p/project1/hai_1191/lukasik1/immu-vis/multiplex-image-model/train_masked_equivariant_config.yaml"
    checkpoint_path = None  # Or specify a checkpoint path
    device = 'cpu'  # Change to 'cuda' if using GPU

    # Load model and data
    model, train_dataloader, test_dataloader, TOKENIZER, INV_TOKENIZER, config = load_model_and_data(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device
    )

    print("\nModel and data loaded successfully!")
    print(f"Tokenizer has {len(TOKENIZER)} markers")
    print(f"Train batches: {len(train_dataloader)}")
    print(f"Test batches: {len(test_dataloader)}")
