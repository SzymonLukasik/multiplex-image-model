#!/usr/bin/env python
"""Smoke test: training val loop + discrete rotation/flip equivariance probe.

Extends smoke_test_trainval.py — same dataloader, same masking, same training
plot — and additionally:

  * For each selected sample and each discrete transform T (90°, 180°, 270°,
    horizontal flip by default), produces a figure laid out exactly like the
    training val plot but with the third column replaced by

         T⁻¹( D( E( T(x_masked) ) ) )

    i.e. the rotated-back reconstruction of the transformed masked input. The
    title above each col-3 image shows the per-channel reconstruction
    consistency for that channel under T: MSE between the reference
    reconstruction D(E(x_masked)) and T⁻¹(D(E(T(x_masked)))).

  * Latent-side equivariance probe. For each (sample, T) emits a figure with
    a few latent channels showing E(x), T⁻¹(E(T(x))), and |diff|. Per-channel
    MSE is shown above col-3. This separates the (verified-perfect) latent
    equivariance from the decoder's residual non-equivariance — and lets us
    eyeball whether the latent stays equivariant at non-training input sizes.

  * Per sample, produces aggregate figures summarising across-channel
    consistency for both the reconstruction and the latent — one bar per
    transform with mean ± std and a per-channel jitter overlay.

  * ``--input-size`` overrides ``config.input_image_size`` so the same model
    can be probed at 113 (the training size) and at 128 (a multiple of 8, no
    post-decoder crop needed) without retraining.

Defaults to D4 minus identity = 90°/180°/270° rotations + a horizontal flip.
rot90 and flips on a square grid are pixel-exact (no resampling), so any
equivariance error you see is the model's.
"""
import argparse
import os
import tempfile
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from math import ceil
from ruamel.yaml import YAML

from load_model_for_evaluation import load_model_and_data
from multiplex_model.utils import plot_reconstructs_with_uncertainty


# ---------------------------------------------------------------------------
# Masking (copied from train_masked_model_ddp_v2.apply_patch_mask)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------
def _to_plain(t):
    """escnn GeometricTensor → torch.Tensor; pass through tensors."""
    return t.tensor if hasattr(t, 'tensor') else t


def _crop_to_input(output: torch.Tensor, input_h: int) -> torch.Tensor:
    """Bring a (B, C, H, W) decoder output down to input resolution.

    When the model upsamples past the input (e.g. 15·8 = 120 from a 113×113
    input), reproduces the training-time ``[3:-4]`` crop convention — drops
    ``(H - input_h) // 2`` from one side and the rest from the other. When
    ``H == input_h`` (e.g. 128 → 128) returns ``output`` unchanged.
    """
    h = output.shape[-2]
    if h == input_h:
        return output
    lo = (h - input_h) // 2
    return output[..., lo:lo + input_h, lo:lo + input_h]


# ---------------------------------------------------------------------------
# Discrete D4 transforms — pixel-exact on square images
# ---------------------------------------------------------------------------
def _apply_transform(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == 'rot90':   return torch.rot90(x, k=1, dims=(-2, -1))
    if name == 'rot180':  return torch.rot90(x, k=2, dims=(-2, -1))
    if name == 'rot270':  return torch.rot90(x, k=3, dims=(-2, -1))
    if name == 'hflip':   return torch.flip(x, dims=(-1,))
    if name == 'vflip':   return torch.flip(x, dims=(-2,))
    raise ValueError(f"Unknown transform: {name!r}")


def _invert_transform(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == 'rot90':   return torch.rot90(x, k=-1, dims=(-2, -1))
    if name == 'rot180':  return torch.rot90(x, k=-2, dims=(-2, -1))
    if name == 'rot270':  return torch.rot90(x, k=-3, dims=(-2, -1))
    if name == 'hflip':   return torch.flip(x, dims=(-1,))   # involution
    if name == 'vflip':   return torch.flip(x, dims=(-2,))
    raise ValueError(f"Unknown transform: {name!r}")


T_TITLES = {
    'rot90':  'T = rot 90°',
    'rot180': 'T = rot 180°',
    'rot270': 'T = rot 270°',
    'hflip':  'T = hflip',
    'vflip':  'T = vflip',
}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_orig_recon_tback(
    orig: torch.Tensor,                   # (B, C, H, W)
    recon_ref: torch.Tensor,              # (B, C, H, W) — D(E(x_masked))
    recon_tback: torch.Tensor,            # (B, C, H, W) — T⁻¹(D(E(T(x_masked))))
    channel_ids: torch.Tensor,            # (B, C)
    masked_ids: List[int],
    markers_names_map: Dict[int, str],
    per_channel_mse: np.ndarray,          # (C,) — MSE(recon_ref, recon_tback) per channel
    suptitle: str,
    ncols: int = 9,
    scale_by_max: bool = True,
):
    """Like plot_reconstructs_with_uncertainty but col3 = T⁻¹∘D∘E∘T, and the
    title above each col3 image carries the per-channel consistency MSE."""
    C = orig.shape[1]
    nrows = ceil(C / (ncols // 3))
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    ax_flat = axs.flatten()
    for i in range(0, len(ax_flat), 3):
        j = i // 3
        ax_img, ax_ref, ax_tb = ax_flat[i], ax_flat[i + 1], ax_flat[i + 2]
        for ax in (ax_img, ax_ref, ax_tb):
            ax.axis('off')

        if j >= C:
            continue

        mid = channel_ids[0, j].item()
        name = markers_names_map.get(mid, f'marker{mid}')

        ax_img.imshow(orig[0, j].cpu().float().numpy(), cmap='CMRmap', vmin=0, vmax=1)
        ax_img.set_title(f'Original\n{name}')

        masked_tag = ' (masked)' if mid in masked_ids else ''
        ax_ref.imshow(recon_ref[0, j].cpu().float().numpy(), cmap='CMRmap', vmin=0, vmax=1)
        ax_ref.set_title(f'Reconstructed{masked_tag}\n{name}')

        if scale_by_max:
            vmax = max(
                float(recon_tback[0, j].max().item()),
                float(recon_ref[0, j].max().item()),
                1e-6,
            )
        else:
            vmax = 1.0
        ax_tb.imshow(recon_tback[0, j].cpu().float().numpy(), cmap='CMRmap',
                     vmin=0, vmax=vmax)
        ax_tb.set_title(f'T⁻¹∘D∘E∘T  MSE={per_channel_mse[j]:.4g}\n{name}')

    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_consistency_aggregate(
    consistency: Dict[int, Dict[str, np.ndarray]],   # sample_idx -> T -> (C,) MSE
    transforms: List[str],
    save_path: str,
    ylabel: str = 'per-channel MSE  (T⁻¹∘D∘E∘T vs D∘E)',
    title_prefix: str = 'across-channel consistency',
):
    """One subplot per sample: bar per transform = mean MSE across channels
    with std error bar; overlaid jitter shows the per-channel distribution."""
    sample_indices = sorted(consistency.keys())
    n = len(sample_indices)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.0), squeeze=False)
    rng = np.random.default_rng(0)
    for i, s in enumerate(sample_indices):
        ax = axes[0][i]
        means, stds = [], []
        for t in transforms:
            vals = consistency[s][t]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
        xs = np.arange(len(transforms))
        ax.bar(xs, means, yerr=stds, color='steelblue', alpha=0.65,
               capsize=4, edgecolor='black')
        # jitter overlay
        for k, t in enumerate(transforms):
            vals = consistency[s][t]
            jitter = rng.normal(loc=k, scale=0.08, size=vals.shape[0])
            ax.scatter(jitter, vals, s=10, alpha=0.5, color='darkorange',
                       edgecolors='none')
        ax.set_xticks(xs)
        ax.set_xticklabels([T_TITLES[t] for t in transforms], rotation=20)
        ax.set_ylabel(ylabel)
        ax.set_title(f'Sample {s}: {title_prefix}')
        ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[smoke-rf] wrote {save_path}")


def _select_latent_channels(latent_np: np.ndarray, k: int) -> List[int]:
    """Pick the k latent channels with the largest L2 norm in E(x_masked)."""
    norms = np.sqrt((latent_np ** 2).reshape(latent_np.shape[0], -1).sum(axis=1))
    k = min(k, latent_np.shape[0])
    return np.argsort(-norms)[:k].tolist()


def plot_latent_equivariance(
    latent_ref: torch.Tensor,         # (C, Hl, Wl) — sample 0 of E(x_masked)
    latent_inv: torch.Tensor,         # (C, Hl, Wl) — T⁻¹(E(T(x_masked)))
    selected_channels: List[int],
    per_channel_mse: np.ndarray,      # (C,) — over ALL latent channels
    suptitle: str,
    save_path: str,
):
    """Per-channel 3-column figure: E(x) | T⁻¹∘E∘T | |diff|. Title above col 3
    carries the per-channel MSE. Channels picked by L2 norm of the reference."""
    nrows = len(selected_channels)
    fig, axes = plt.subplots(nrows, 3, figsize=(8.0, 2.5 * nrows), squeeze=False)
    ref = latent_ref.cpu().float().numpy()
    inv = latent_inv.cpu().float().numpy()
    diff = np.abs(ref - inv)
    for r, ch in enumerate(selected_channels):
        ax_ref, ax_inv, ax_diff = axes[r]
        # Share scale between ref and inv per row so visual comparison is fair.
        vmax_ri = max(float(np.abs(ref[ch]).max()), float(np.abs(inv[ch]).max()), 1e-6)
        vmax_d = max(float(diff[ch].max()), 1e-6)
        ax_ref.imshow(ref[ch], cmap='RdBu_r', vmin=-vmax_ri, vmax=vmax_ri)
        ax_inv.imshow(inv[ch], cmap='RdBu_r', vmin=-vmax_ri, vmax=vmax_ri)
        ax_diff.imshow(diff[ch], cmap='magma', vmin=0, vmax=vmax_d)
        for ax in (ax_ref, ax_inv, ax_diff):
            ax.set_xticks([]); ax.set_yticks([])
        ax_ref.set_ylabel(f'latent ch {ch}', fontsize=9)
        if r == 0:
            ax_ref.set_title('E(x_masked)')
            ax_inv.set_title('T⁻¹(E(T(x_masked)))')
            ax_diff.set_title('|diff|')
        ax_diff.text(
            0.5, 1.02, f'MSE={per_channel_mse[ch]:.3e}',
            transform=ax_diff.transAxes, ha='center', va='bottom', fontsize=9,
        )
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[smoke-rf] wrote {save_path}")


# ---------------------------------------------------------------------------
# Args / config helpers
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--model-type', default=None,
                   help='Override config.model_type (rarely needed)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output-dir', default='smoke_test_trainval_rf')
    p.add_argument('--num-plots', type=int, default=5,
                   help='Number of val batches to plot')
    p.add_argument('--transforms', nargs='*',
                   default=['rot90', 'rot180', 'rot270', 'hflip'],
                   choices=['rot90', 'rot180', 'rot270', 'hflip', 'vflip'],
                   help='Discrete D4 transforms to probe (default: D4 minus '
                        'identity, minus vflip).')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--spatial-masking-ratio', type=float, default=None)
    p.add_argument('--fully-masked-channels-max-frac', type=float, default=None)
    p.add_argument('--mask-patch-size', type=int, default=None)
    p.add_argument('--input-size', type=int, default=None,
                   help='Override config.input_image_size. Use 128 to bypass '
                        'the asymmetric crop (15·8=120 → cropped to 113 vs '
                        '16·8=128 → no crop), or any other size to probe '
                        'behaviour off-distribution from training.')
    p.add_argument('--num-latent-channels', type=int, default=6,
                   help='How many latent channels (selected by L2 norm of '
                        'E(x_masked)) to visualise per latent figure.')
    return p.parse_args()


def _config_with_overrides(config_path, model_type=None, input_size=None):
    """Write a temp config with optional model_type / input_image_size overrides."""
    if not model_type and not input_size:
        return config_path, None
    yaml = YAML(typ='safe')
    with open(config_path) as f:
        cfg = yaml.load(f)
    if model_type:
        cfg['model_type'] = model_type
    if input_size:
        cfg['input_image_size'] = [int(input_size), int(input_size)]
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.yaml', delete=False, prefix='smoke_cfg_'
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    print(f"[smoke-rf] overrides applied: model_type={model_type!r}, "
          f"input_size={input_size!r} (temp config: {tmp.name})")
    return tmp.name, tmp.name


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------
@torch.no_grad()
def _forward_full(model, masked_img, active_channel_ids, channel_ids):
    """Train-val forward path: full reconstruction + final latent.

    Decoder output is cropped to the input resolution dynamically (no crop
    needed when input is already a multiple of 8). Latent (last entry of
    ``features``) is converted to a plain tensor.
    """
    out = model(masked_img, active_channel_ids, channel_ids, True)
    output = _crop_to_input(out['output'], masked_img.shape[-2])
    mi, logvar = output.unbind(dim=-1)
    latent = _to_plain(out['features'][-1]).float()
    return (
        torch.sigmoid(mi),
        torch.clamp(logvar, min=-15.0, max=15.0),
        latent,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg_path, tmp_cfg = _config_with_overrides(
        args.config, model_type=args.model_type, input_size=args.input_size,
    )
    try:
        model, _, test_dataloader, _tok, inv_tokenizer, config = load_model_and_data(
            config_path=cfg_path, checkpoint_path=args.checkpoint, device=args.device,
        )
    finally:
        if tmp_cfg and os.path.exists(tmp_cfg):
            os.unlink(tmp_cfg)
    model.eval()
    print(f"[smoke-rf] effective input_image_size = {config['input_image_size']}")

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
    print(f"[smoke-rf] masking: spatial_ratio={spatial_masking_ratio}, "
          f"fully_masked_max_frac={fully_masked_channels_max_frac}, "
          f"patch_size={mask_patch_size}")
    print(f"[smoke-rf] transforms: {args.transforms}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[smoke-rf] output_dir = {args.output_dir}")

    num_plots = min(args.num_plots, len(test_dataloader))
    plot_indices = set(np.random.choice(
        np.arange(len(test_dataloader)), size=num_plots, replace=False,
    ).tolist())
    max_idx = max(plot_indices)
    print(f"[smoke-rf] plotting {num_plots} batch(es): "
          f"indices={sorted(plot_indices)}")

    # sample_idx (= batch idx) -> transform_name -> per-channel MSE (C,)
    consistency: Dict[int, Dict[str, np.ndarray]] = {}
    latent_consistency: Dict[int, Dict[str, np.ndarray]] = {}

    with torch.no_grad():
        for idx, (img, channel_ids, _panel_idx, _img_path) in enumerate(test_dataloader):
            if idx > max_idx:
                break
            if idx not in plot_indices:
                continue

            # -------- training-val masking (mask ONCE; reused for all T's) --
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

            # -------- reference forward (= the training val plot) ----------
            mi_ref, logvar, lat_ref = _forward_full(
                model, masked_img, active_channel_ids, channel_ids,
            )
            uncertainty_img = torch.exp(logvar / 2)
            # Pick latent channels by L2 norm on sample 0 of the reference latent.
            lat_ref_np = lat_ref[0].cpu().float().numpy()
            picked_latent_channels = _select_latent_channels(
                lat_ref_np, args.num_latent_channels,
            )
            unactive_channels = [i for i in channel_ids[0] if i not in active_channel_ids[0]]
            unactive_ids_int = [int(c.item()) for c in unactive_channels]

            fig = plot_reconstructs_with_uncertainty(
                img.float(), mi_ref.float(), uncertainty_img.float(),
                channel_ids, unactive_channels,
                markers_names_map=inv_tokenizer, scale_by_max=True,
            )
            ref_path = os.path.join(args.output_dir, f'val_batch{idx:05d}_ref.png')
            fig.savefig(ref_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f"[smoke-rf] wrote {ref_path}")

            # -------- one figure per transform -----------------------------
            consistency[idx] = {}
            latent_consistency[idx] = {}
            for t in args.transforms:
                x_T = _apply_transform(masked_img, t)
                mi_T, _, lat_T = _forward_full(
                    model, x_T, active_channel_ids, channel_ids,
                )
                mi_T_back = _invert_transform(mi_T, t)
                lat_T_back = _invert_transform(lat_T, t)

                # Per-channel consistency MSE (B=1: first sample). For the
                # whole batch we'd average over B, but matches the training
                # plot convention which only displays the first sample.
                diff = (mi_T_back[0] - mi_ref[0]) ** 2          # (C, H, W)
                per_ch_mse = diff.mean(dim=(-2, -1)).cpu().float().numpy()  # (C,)
                consistency[idx][t] = per_ch_mse

                lat_diff = (lat_T_back[0] - lat_ref[0]) ** 2     # (Cz, Hl, Wl)
                lat_per_ch_mse = lat_diff.mean(dim=(-2, -1)).cpu().float().numpy()
                latent_consistency[idx][t] = lat_per_ch_mse

                fig = plot_orig_recon_tback(
                    orig=img, recon_ref=mi_ref, recon_tback=mi_T_back,
                    channel_ids=channel_ids,
                    masked_ids=unactive_ids_int,
                    markers_names_map=inv_tokenizer,
                    per_channel_mse=per_ch_mse,
                    suptitle=(f'Batch {idx} — {T_TITLES[t]}  '
                              f'(across-channel mean MSE = {per_ch_mse.mean():.4g})'),
                )
                save_path = os.path.join(
                    args.output_dir, f'val_batch{idx:05d}_{t}.png'
                )
                fig.savefig(save_path, dpi=120, bbox_inches='tight')
                plt.close(fig)
                print(f"[smoke-rf] wrote {save_path}")

                plot_latent_equivariance(
                    latent_ref=lat_ref[0],
                    latent_inv=lat_T_back[0],
                    selected_channels=picked_latent_channels,
                    per_channel_mse=lat_per_ch_mse,
                    suptitle=(f'Batch {idx} — {T_TITLES[t]}  latent equivariance '
                              f'(across-channel mean MSE = {lat_per_ch_mse.mean():.3e})'),
                    save_path=os.path.join(
                        args.output_dir, f'val_batch{idx:05d}_{t}_latent.png',
                    ),
                )

    if consistency:
        plot_consistency_aggregate(
            consistency=consistency, transforms=args.transforms,
            save_path=os.path.join(args.output_dir, 'aggregate_consistency.png'),
        )
    if latent_consistency:
        plot_consistency_aggregate(
            consistency=latent_consistency, transforms=args.transforms,
            save_path=os.path.join(args.output_dir, 'aggregate_latent_consistency.png'),
            ylabel='per-channel MSE  (T⁻¹∘E∘T vs E)',
            title_prefix='latent equivariance (across-channel)',
        )
    print(f"[smoke-rf] done.")


if __name__ == '__main__':
    main()
