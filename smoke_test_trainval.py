#!/usr/bin/env python
"""Smoke test using the training val loop's dataloader + plotting function.

Mirrors ``test_masked`` from train_masked_model_ddp_v2.py:
  * same dataloader (load_model_and_data → DatasetFromTIFF + TestCrop, identical
    to the training script's test_dataset),
  * same masking (random channel drop + spatial patch mask) using the masking
    knobs from the config,
  * same decoder-output crop ``[:, :, 3:-4, 3:-4]``,
  * same plotting function ``plot_reconstructs_with_uncertainty`` (Original /
    Reconstructed (masked/partially masked tag) / Variance triplets, scaled
    per-channel), invoked with inv_tokenizer as ``markers_names_map``.

Difference vs training: figures are written to disk instead of being logged to
TensorBoard, and DDP/all_reduce is skipped (single-process, single-rank).
"""
import argparse
import os
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from ruamel.yaml import YAML

from load_model_for_evaluation import load_model_and_data
from multiplex_model.utils import plot_reconstructs_with_uncertainty


# Copied verbatim from train_masked_model_ddp_v2.apply_patch_mask so we don't
# pull DDP/training-time imports just for this helper.
def apply_patch_mask(x: torch.Tensor, ratio: float, patch_size: int) -> torch.Tensor:
    B, C, H, W = x.shape
    pad_h = (patch_size - (H % patch_size)) % patch_size
    pad_w = (patch_size - (W % patch_size)) % patch_size
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)
    Hp, Wp = x.shape[-2:]
    h, w = Hp // patch_size, Wp // patch_size
    total_patches = h * w
    patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).contiguous()
    patches = patches.view(B, C, total_patches, patch_size * patch_size)
    mask = torch.rand((B, C, total_patches), device=x.device) < ratio
    patches[mask] = 0.0
    x = patches.view(B, C, h, w, patch_size, patch_size)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, Hp, Wp)
    return x[..., :H, :W]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--model-type', default=None,
                   help='Override config.model_type (rarely needed)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output-dir', default='smoke_test_trainval')
    p.add_argument('--num-plots', type=int, default=5,
                   help='Number of val batches to plot (one figure per batch, '
                        'first sample of the batch — same convention as test_masked).')
    p.add_argument('--seed', type=int, default=42)
    # Masking knobs: default to the config's values (matches training-val);
    # override only to probe different mask regimes.
    p.add_argument('--spatial-masking-ratio', type=float, default=None)
    p.add_argument('--fully-masked-channels-max-frac', type=float, default=None)
    p.add_argument('--mask-patch-size', type=int, default=None)
    return p.parse_args()


def _config_with_model_type_override(config_path, model_type):
    if not model_type:
        return config_path, None
    yaml = YAML(typ='safe')
    with open(config_path) as f:
        cfg = yaml.load(f)
    cfg['model_type'] = model_type
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.yaml', delete=False, prefix='smoke_cfg_'
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    print(f"[smoke-trainval] model_type overridden to '{model_type}' "
          f"(temp config: {tmp.name})")
    return tmp.name, tmp.name


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg_path, tmp_cfg = _config_with_model_type_override(args.config, args.model_type)
    try:
        model, _, test_dataloader, _tok, inv_tokenizer, config = load_model_and_data(
            config_path=cfg_path, checkpoint_path=args.checkpoint, device=args.device,
        )
    finally:
        if tmp_cfg and os.path.exists(tmp_cfg):
            os.unlink(tmp_cfg)
    model.eval()

    # Masking knobs default to what training/val used (per the config).
    spatial_masking_ratio = (
        args.spatial_masking_ratio
        if args.spatial_masking_ratio is not None
        else config.get('spatial_masking_ratio', 0.6)
    )
    fully_masked_channels_max_frac = (
        args.fully_masked_channels_max_frac
        if args.fully_masked_channels_max_frac is not None
        else config.get('fully_masked_channels_max_frac', 0.5)
    )
    mask_patch_size = (
        args.mask_patch_size
        if args.mask_patch_size is not None
        else config.get('mask_patch_size', 8)
    )
    print(f"[smoke-trainval] masking: spatial_ratio={spatial_masking_ratio}, "
          f"fully_masked_max_frac={fully_masked_channels_max_frac}, "
          f"patch_size={mask_patch_size}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[smoke-trainval] output_dir = {args.output_dir}")

    num_plots = min(args.num_plots, len(test_dataloader))
    plot_indices = set(np.random.choice(
        np.arange(len(test_dataloader)), size=num_plots, replace=False,
    ).tolist())
    max_idx = max(plot_indices)
    print(f"[smoke-trainval] plotting {num_plots} batch(es): "
          f"indices={sorted(plot_indices)}")

    with torch.no_grad():
        for idx, (img, channel_ids, _panel_idx, _img_path) in enumerate(test_dataloader):
            if idx > max_idx:
                break
            if idx not in plot_indices:
                continue

            # ---------- training-val masking (verbatim from test_masked) ----
            batch_size, num_channels, H, W = img.shape
            img = img.to(args.device, dtype=torch.float32)
            channel_ids = channel_ids.to(args.device, dtype=torch.long)

            max_channels_to_mask = int(np.ceil(num_channels * fully_masked_channels_max_frac))
            num_channels_to_mask = np.random.randint(1, max_channels_to_mask + 1)
            masked_img, active_channel_ids = [], []
            for b_i in range(batch_size):
                channels_to_keep = torch.randperm(num_channels)[num_channels_to_mask:]
                masked_img.append(img[b_i:b_i + 1, channels_to_keep, :, :])
                active_channel_ids.append(channel_ids[b_i:b_i + 1, channels_to_keep])
            masked_img = torch.cat(masked_img, dim=0).to(args.device, dtype=torch.float32)
            active_channel_ids = torch.cat(active_channel_ids, dim=0).to(args.device)
            masked_img = apply_patch_mask(masked_img, spatial_masking_ratio, mask_patch_size)

            # ---------- forward (same crop convention as test_masked) -------
            output = model(masked_img, active_channel_ids, channel_ids)['output'][:, :, 3:-4, 3:-4]
            mi, logvar = output.unbind(dim=-1)
            mi = torch.sigmoid(mi)
            logvar = torch.clamp(logvar, min=-15.0, max=15.0)
            uncertainty_img = torch.exp(logvar / 2)

            # ---------- plot via the training plotter ----------------------
            unactive_channels = [i for i in channel_ids[0] if i not in active_channel_ids[0]]
            fig = plot_reconstructs_with_uncertainty(
                img.float(),
                mi.float(),
                uncertainty_img.float(),
                channel_ids,
                unactive_channels,
                markers_names_map=inv_tokenizer,
                scale_by_max=True,
            )
            save_path = os.path.join(args.output_dir, f'val_batch{idx:05d}.png')
            fig.savefig(save_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f"[smoke-trainval] wrote {save_path}")

    print(f"[smoke-trainval] done.")


if __name__ == '__main__':
    main()
