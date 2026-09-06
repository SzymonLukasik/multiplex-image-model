"""Leave-one-out inpainting validation for the *fresh-repo* models.

This is a port of the main repo's ``run_validation_equivariant.py`` to the
package API used here. The scoring protocol — leave-one-out channel holdout,
per-observation MSE / Pearson / logsigma, dark/bright stratification, and the
sufficient statistics for globally-pooled per-marker metrics — is carried over
unchanged so the CSVs have the same schema and the same meaning.

What differs from the main-repo script, and why:

* **Model construction** goes through ``train_masked_model.build_model`` /
  ``load_model_from_checkpoint`` instead of the hand-rolled per-model-type
  dispatch. That is the only way to load the models trained from this repo:
  the progressive decoder (``upsample_type: progressive``) exists only here, so
  its ``state_dict`` has keys no main-repo class defines.
* **Architecture comes from the checkpoint** when it carries a ``model_config``
  (all checkpoints written by this repo do). The ``--config`` YAML is then used
  only for the data pipeline. This makes an eval immune to config drift — you
  cannot accidentally score a checkpoint against a differently-shaped model.
* **No output cropping.** These models take a 128x128 input, downscale to
  ``internal_image_size`` internally and upscale the reconstruction back, so the
  output is already aligned with the target. The shapes are asserted rather than
  assumed.
* **``--internal-resolution``** scores at the model's internal size instead: the
  final upscale is bypassed and the target is downscaled with the same bilinear
  op the model applies to its input. This removes the resampling from the metric,
  which otherwise both denies the model credit for detail it cannot represent and
  attenuates the amplitude of what it does produce.

Note on comparability: results are at the 128x128 input size these models train
at, NOT the crop 80 / crop 112 of the older main-repo CSVs. Runs produced here
are comparable to each other, not to those.

Usage:
    python run_validation.py \
        --config configs/train_vanilla_wide_progdec_zs_hn_config.yaml \
        --checkpoint /path/to/checkpoint-...-epoch_29.pth \
        --panel-config configs/all_panels_config_tscratch_hn_only.yaml \
        --all-channels-loo \
        --output-dir validation_results_zs_hn_...
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from multiplex_model.data import MultiplexDataset, TestCrop
from multiplex_model.utils.configuration import TrainingConfig
from train_masked_model import build_model, load_model_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Leave-one-out inpainting validation for fresh-repo models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Training config YAML (used for the data pipeline; the "
                             "architecture is taken from the checkpoint when available).")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint (.pth).")
    parser.add_argument("--model-type", type=str, default="auto",
                        choices=["auto", "vanilla", "fully_equivariant_v3"],
                        help="'auto' reads model_type from the config.")
    parser.add_argument("--panel-config", type=str, default=None,
                        help="Override the config's panel_config, e.g. the hn-only "
                             "panel for zero-shot evaluation.")
    parser.add_argument("--masked-channels", type=int, nargs="+", default=[1],
                        help="Numbers of channels to hold out together. Ignored with "
                             "--all-channels-loo.")
    parser.add_argument("--random-samples-per-image", type=int, default=32,
                        help="Random mask draws per image. Ignored with --all-channels-loo.")
    parser.add_argument("--all-channels-loo", action="store_true",
                        help="Deterministic leave-one-out: hold out EACH channel exactly "
                             "once per image (C observations/image, full reproducible "
                             "coverage).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="validation_results")
    parser.add_argument("--dark-threshold", type=float, default=0.05,
                        help="Intensity threshold for dark regions (0-1 scale).")
    parser.add_argument("--forward-batch-size", type=int, default=4,
                        help="Sub-batch size for the per-image forward pass. Caps peak "
                             "GPU memory; the escnn model needs this small.")
    parser.add_argument("--patch-mask-ratio", type=float, default=0.0,
                        help="Fraction of spatial patches zeroed on the KEPT (input) "
                             "channels, matching training-time spatial masking (these "
                             "models train at 0.6). 0.0 = fully-dense input, which is "
                             "out-of-distribution for them. Does NOT affect the target.")
    parser.add_argument("--mask-patch-size", type=int, default=8,
                        help="Patch size for --patch-mask-ratio (train config uses 8).")
    parser.add_argument("--internal-resolution", action="store_true",
                        help="Score the reconstruction at the model's INTERNAL "
                             "resolution instead of the input resolution. These "
                             "models bilinearly downscale 128 -> internal_image_size, "
                             "reconstruct there, and upscale back; the default "
                             "protocol therefore scores them against detail they "
                             "cannot represent, and the upscale attenuates whatever "
                             "they do produce. With this flag the final upscale is "
                             "bypassed and the target is downscaled the same way the "
                             "model downscales its input, so the metric reflects the "
                             "reconstruction alone. Not comparable to runs without it.")
    parser.add_argument("--limit-images", type=int, default=None,
                        help="Evaluate only the first N test images. For smoke-testing "
                             "the pipeline on a compute node before the full run.")
    parser.add_argument("--save-maps-path", type=str, default=None,
                        help="If set, dump per-pixel predicted mean (post-sigmoid), "
                             "predicted logvar and ground truth for every held-out "
                             "prediction to this .npz, so figures and pooled metrics can "
                             "be recomputed offline. hn-scale only.")
    return parser.parse_args()


def apply_patch_mask(x: torch.Tensor, ratio: float, patch_size: int) -> torch.Tensor:
    """Zero out a fraction of spatial patches, independently per (sample, channel).

    Kept identical to the training-time spatial masking so the eval can reproduce
    the exact input distribution the model was trained under. With ratio=0.0 this
    is a no-op.
    """
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


def create_leave_one_out_batch(img, channel_ids):
    """Deterministic leave-one-out: one variant per channel, that channel held out."""
    C, H, W = img.shape
    masked_indices = torch.arange(C, device=img.device, dtype=torch.long).unsqueeze(1)  # [C, 1]

    keep_mask = torch.ones((C, C), dtype=torch.bool, device=img.device)
    keep_mask.scatter_(1, masked_indices, False)  # row i keeps every channel but i

    img_expanded = img.unsqueeze(0).expand(C, -1, -1, -1)
    channel_ids_expanded = channel_ids.unsqueeze(0).expand(C, -1)

    perturbed_batch = img_expanded[keep_mask].view(C, C - 1, H, W)
    input_channel_ids = channel_ids_expanded[keep_mask].view(C, C - 1)
    output_channel_ids = channel_ids[masked_indices]      # [C, 1]
    target_channels = img[masked_indices]                 # [C, 1, H, W]
    return perturbed_batch, input_channel_ids, output_channel_ids, masked_indices, target_channels


def create_random_channel_dropout_batch(img, channel_ids, num_masked_channels, num_samples, rng):
    """Random channel dropout: `num_samples` draws of `num_masked_channels` held out."""
    C, H, W = img.shape
    if num_masked_channels < 1 or num_masked_channels >= C:
        raise ValueError(f"num_masked_channels must be in [1, {C-1}], got {num_masked_channels}")

    masked_indices_np = np.stack(
        [
            np.sort(rng.choice(C, size=num_masked_channels, replace=False))
            for _ in range(num_samples)
        ],
        axis=0,
    )
    masked_indices = torch.from_numpy(masked_indices_np).to(img.device, dtype=torch.long)

    num_samples = masked_indices.shape[0]
    keep_mask = torch.ones((num_samples, C), dtype=torch.bool, device=img.device)
    keep_mask.scatter_(1, masked_indices, False)

    input_channels_count = C - num_masked_channels
    img_expanded = img.unsqueeze(0).expand(num_samples, -1, -1, -1)
    channel_ids_expanded = channel_ids.unsqueeze(0).expand(num_samples, -1)

    perturbed_batch = img_expanded[keep_mask].view(num_samples, input_channels_count, H, W)
    input_channel_ids = channel_ids_expanded[keep_mask].view(num_samples, input_channels_count)
    output_channel_ids = channel_ids[masked_indices]
    target_channels = img[masked_indices]

    return perturbed_batch, input_channel_ids, output_channel_ids, masked_indices, target_channels


def compute_dark_bright_metrics(mi, target, dark_mask, bright_mask):
    """MSE and Pearson restricted to dark and to bright pixels."""
    metrics = {}

    for label, mask in (("dark", dark_mask), ("bright", bright_mask)):
        if not mask.any():
            continue
        mi_r = mi[:, :, mask]
        target_r = target[:, :, mask]

        mse_r = (mi_r - target_r).pow(2).mean()
        mi_mean_r = mi_r.mean(dim=-1, keepdim=True)
        target_mean_r = target_r.mean(dim=-1, keepdim=True)

        if target_r.std() > 1e-8 and mi_r.std() > 1e-8:
            pearson_r = ((mi_r - mi_mean_r) * (target_r - target_mean_r)).mean(dim=-1) / (
                mi_r.std(dim=-1) * target_r.std(dim=-1) + 1e-8
            )
        else:
            pearson_r = torch.full((mi.shape[0],), np.nan, device=mi.device)

        metrics[label] = {"mse": mse_r.item(), "pearson": pearson_r.mean().item()}

    return metrics


def load_eval_model(args, config, num_channels, device):
    """Build the model, preferring the architecture stored in the checkpoint.

    Checkpoints written by this repo carry ``model_config``; using it means the
    eval cannot silently score a checkpoint against a differently-shaped model
    built from a drifted YAML. Falls back to the config for older checkpoints.
    """
    model_type = config.model_type if args.model_type == "auto" else args.model_type
    print(f"Model type: {model_type}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    epoch = checkpoint.get("epoch", "unknown")
    print(f"Checkpoint epoch: {epoch}")

    model_config = checkpoint.get("model_config")
    if model_config is not None:
        print("Architecture: taken from the checkpoint's model_config")
    else:
        print("Architecture: checkpoint has no model_config - falling back to --config")
        model_config = {
            "num_channels": num_channels,
            "encoder_config": config.encoder_config.model_dump(),
            "decoder_config": config.decoder_config.model_dump(),
        }

    if model_config["num_channels"] != num_channels:
        raise ValueError(
            f"Tokenizer has {num_channels} markers but the checkpoint was trained with "
            f"{model_config['num_channels']}. Wrong tokenizer_config for this checkpoint."
        )

    model = load_model_from_checkpoint(checkpoint, model_config, model_type).to(device)
    model.eval()
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model, model_type, epoch


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    yaml = YAML(typ="safe")
    with open(args.config, "r") as f:
        raw_config = yaml.load(f)
    if args.panel_config is not None:
        print(f"Panel config overridden: {args.panel_config}")
        raw_config["panel_config"] = args.panel_config
    config = TrainingConfig(**raw_config)

    tokenizer = config.tokenizer_config
    inv_tokenizer = {v: k for k, v in tokenizer.items()}
    num_channels = len(tokenizer)
    crop_size = config.input_image_size[0]

    print(f"Number of markers in tokenizer: {num_channels}")
    print(f"Input crop size: {crop_size}")
    if args.all_channels_loo:
        print("Holdout protocol: DETERMINISTIC all-channel leave-one-out "
              "(each channel held out once per image; --random-samples-per-image ignored)")
    else:
        print(f"Holdout protocol: {args.random_samples_per_image} random draws/image, "
              f"masked-channel settings {args.masked_channels}")
    print(f"Dark threshold: {args.dark_threshold * 100:.1f}%")
    if args.patch_mask_ratio > 0:
        print(f"Patch masking on kept channels: ratio={args.patch_mask_ratio}, "
              f"patch_size={args.mask_patch_size} (matches training distribution)")
    else:
        print("Patch masking on kept channels: DISABLED (fully-dense input, OOD for "
              "models trained with spatial_masking_ratio 0.6)")
    if args.save_maps_path:
        print(f"Saving per-pixel mean/logvar/target maps to: {args.save_maps_path}")

    model, model_type, ckpt_epoch = load_eval_model(args, config, num_channels, device)

    # Optionally score at the model's internal resolution. Both the vanilla and the
    # equivariant autoencoder expose the same _downscale/_upscale pair, so bypassing
    # the final upscale is one assignment; the target is then resampled with the
    # SAME op the model applies to its input (bilinear, align_corners=False), so the
    # two sides of the metric stay aligned.
    internal_size = getattr(model, "_internal_size", None)
    eval_size = crop_size
    if args.internal_resolution:
        if internal_size is None:
            raise ValueError(
                "--internal-resolution requires a model with internal_image_size set; "
                f"this checkpoint has none (it already reconstructs at {crop_size})."
            )
        model._upscale = lambda recon: recon
        eval_size = internal_size
        print(f"Scoring at INTERNAL resolution {internal_size} "
              f"(final {internal_size}->{crop_size} upscale bypassed; "
              f"target downscaled to match)")
    else:
        print(f"Scoring at INPUT resolution {crop_size}"
              + (f" (model reconstructs at {internal_size} and upscales)"
                 if internal_size else ""))

    def to_eval_res(t):
        """Bilinearly resample [N, 1, H, W] to the scoring resolution."""
        if t.shape[-1] == eval_size:
            return t
        return F.interpolate(t, size=(eval_size, eval_size),
                             mode="bilinear", align_corners=False)

    test_dataset = MultiplexDataset(
        panels_config=config.panel_config,
        split="test",
        marker_tokenizer=tokenizer,
        transform=TestCrop(crop_size),
        **config.data_config.model_dump(),
    )
    print(f"Test dataset size: {len(test_dataset)} images")

    # batch_size=1: every image is expanded into its own leave-one-out batch, so
    # no panel-homogeneous sampler is needed here.
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)  # makes the optional patch mask reproducible

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    save_maps = args.save_maps_path is not None
    map_store = ({"pred_mean": [], "pred_logvar": [], "target": [],
                  "marker": [], "channel_id": [], "dataset_name": [], "image_path": []}
                 if save_maps else None)

    with torch.no_grad():
        for n_done, (img, channel_ids, ds_name, img_path) in enumerate(
            tqdm(dataloader, desc="Evaluating on test set")
        ):
            if args.limit_images is not None and n_done >= args.limit_images:
                print(f"Stopping after --limit-images {args.limit_images} images")
                break
            img = img.squeeze(0).to(device, dtype=torch.float32)
            channel_ids = channel_ids.squeeze(0).to(device, dtype=torch.long)

            C, H, W = img.shape
            valid_masked_channels = [k for k in args.masked_channels if 1 <= k < C]
            if args.all_channels_loo:
                valid_masked_channels = [1]  # LOO is single-channel; run exactly once

            # Dark/bright stratification uses the mean intensity across all channels,
            # computed at whatever resolution the metrics are scored at.
            intensity_src = to_eval_res(img.mean(dim=0)[None, None])[0, 0]
            intensity_map = intensity_src  # [h, w]
            intensity_min, intensity_max = intensity_map.min(), intensity_map.max()
            normalized_intensity = (intensity_map - intensity_min) / (intensity_max - intensity_min + 1e-8)
            dark_mask = normalized_intensity < args.dark_threshold
            bright_mask = normalized_intensity >= args.dark_threshold

            for masked_channels_count in valid_masked_channels:
                if args.all_channels_loo:
                    (
                        perturbed_batch,
                        input_channel_ids,
                        output_channel_ids,
                        masked_indices,
                        target_channels,
                    ) = create_leave_one_out_batch(img, channel_ids)
                    masked_channels_count = 1
                else:
                    (
                        perturbed_batch,
                        input_channel_ids,
                        output_channel_ids,
                        masked_indices,
                        target_channels,
                    ) = create_random_channel_dropout_batch(
                        img=img,
                        channel_ids=channel_ids,
                        num_masked_channels=masked_channels_count,
                        num_samples=args.random_samples_per_image,
                        rng=rng,
                    )

                # Spatially patch-mask the KEPT (input) channels only. The held-out
                # target is untouched, so the scoring protocol is unchanged - only
                # the input statistics move toward the training distribution.
                if args.patch_mask_ratio > 0:
                    perturbed_batch = apply_patch_mask(
                        perturbed_batch, args.patch_mask_ratio, args.mask_patch_size
                    )

                # Chunk the per-image variants through the model to cap peak GPU
                # memory: escnn layers build large intermediates that scale with
                # batch size, so all C variants at once OOMs even though training
                # at batch 4 fits.
                chunk = max(1, args.forward_batch_size)
                output_chunks = []
                for start in range(0, perturbed_batch.shape[0], chunk):
                    end = start + chunk
                    out_chunk = model(
                        perturbed_batch[start:end],
                        input_channel_ids[start:end],
                        output_channel_ids[start:end],
                    )["output"]
                    output_chunks.append(out_chunk)
                output = torch.cat(output_chunks, dim=0)

                mi, logvar = output.unbind(dim=-1)
                mi = torch.sigmoid(mi)

                # Bring the target to the scoring resolution. This is a no-op in the
                # default protocol (the model already upscaled to the input size).
                target_channels = to_eval_res(target_channels)

                # Assert rather than assume - a silent mismatch here would corrupt
                # every metric below.
                if mi.shape != target_channels.shape:
                    raise RuntimeError(
                        f"Output shape {tuple(mi.shape)} != target shape "
                        f"{tuple(target_channels.shape)} for model_type={model_type}."
                    )

                # Overall metrics
                mse_overall = (mi - target_channels).pow(2).mean(dim=(2, 3))
                mi_mean = mi.mean(dim=(2, 3), keepdim=True)
                target_mean = target_channels.mean(dim=(2, 3), keepdim=True)
                pearson_overall = ((mi - mi_mean) * (target_channels - target_mean)).mean(dim=(2, 3)) / (
                    mi.std(dim=(2, 3)) * target_channels.std(dim=(2, 3)) + 1e-8
                )
                logsigma_overall = logvar.mean(dim=(2, 3))

                # Sufficient statistics for GLOBALLY-POOLED per-marker metrics.
                # The per-observation pearson/mse above are per-image; averaging
                # them weights every patch equally. Pooling all pixels of a marker
                # across images needs additive sums, not averaged scalars.
                n_pix = mi.shape[2] * mi.shape[3]
                sum_pred = mi.sum(dim=(2, 3))
                sum_gt = target_channels.sum(dim=(2, 3))
                sum_pred2 = (mi * mi).sum(dim=(2, 3))
                sum_gt2 = (target_channels * target_channels).sum(dim=(2, 3))
                sum_predgt = (mi * target_channels).sum(dim=(2, 3))
                sum_se = (mi - target_channels).pow(2).sum(dim=(2, 3))

                region_metrics = compute_dark_bright_metrics(
                    mi, target_channels, dark_mask, bright_mask
                )

                num_observations = masked_indices.numel()
                for i in range(num_observations):
                    cid = channel_ids[masked_indices[i]].item()
                    result = {
                        "mse": mse_overall[i].item(),
                        "logsigma": logsigma_overall[i].item(),
                        "pearson": pearson_overall[i].item(),
                        "Channel_ID": cid,
                        "marker": inv_tokenizer.get(cid, "Unknown"),
                        "masked_count": masked_channels_count,
                        "dataset_name": ds_name[0],
                        "image_path": img_path[0],
                        "dark_fraction": dark_mask.sum().item() / dark_mask.numel(),
                        "sum_pred": sum_pred[i].item(),
                        "sum_gt": sum_gt[i].item(),
                        "sum_pred2": sum_pred2[i].item(),
                        "sum_gt2": sum_gt2[i].item(),
                        "sum_predgt": sum_predgt[i].item(),
                        "sum_se": sum_se[i].item(),
                        "n_pix": n_pix,
                    }
                    for region, metrics in region_metrics.items():
                        result[f"mse_{region}"] = metrics["mse"]
                        result[f"pearson_{region}"] = metrics["pearson"]

                    all_results.append(result)

                    if save_maps:
                        map_store["pred_mean"].append(mi[i, 0].detach().cpu().numpy().astype(np.float32))
                        map_store["pred_logvar"].append(logvar[i, 0].detach().cpu().numpy().astype(np.float32))
                        map_store["target"].append(target_channels[i, 0].detach().cpu().numpy().astype(np.float32))
                        map_store["marker"].append(inv_tokenizer.get(cid, "Unknown"))
                        map_store["channel_id"].append(cid)
                        map_store["dataset_name"].append(ds_name[0])
                        map_store["image_path"].append(img_path[0])

    df = pd.DataFrame(all_results)

    checkpoint_name_clean = os.path.splitext(os.path.basename(args.checkpoint))[0]
    df["model"] = checkpoint_name_clean
    df["masked"] = "masked"
    df["ckpt_epoch"] = ckpt_epoch
    df["patch_mask_ratio"] = args.patch_mask_ratio
    df["crop_size"] = crop_size
    df["eval_size"] = eval_size
    df["scored_at"] = "internal" if args.internal_resolution else "input"

    cols = ["mse", "logsigma", "pearson", "Channel_ID", "marker", "masked", "masked_count",
            "dataset_name", "image_path", "model", "ckpt_epoch", "patch_mask_ratio",
            "crop_size", "eval_size", "scored_at", "dark_fraction"]
    cols = [c for c in cols if c in df.columns]
    cols += [c for c in df.columns
             if (c.startswith("mse_") or c.startswith("pearson_")) and c not in cols]
    cols += [c for c in ["sum_pred", "sum_gt", "sum_pred2", "sum_gt2",
                         "sum_predgt", "sum_se", "n_pix"] if c in df.columns]
    df = df[cols]

    if args.masked_channels == [1] or args.all_channels_loo:
        output_file = os.path.join(args.output_dir, f"{checkpoint_name_clean}_full.csv")
    else:
        mask_tag = "-".join(map(str, args.masked_channels))
        output_file = os.path.join(args.output_dir, f"{checkpoint_name_clean}_full_k{mask_tag}.csv")

    print(f"Saving results to: {output_file}")
    df.to_csv(output_file, index=False)
    print(f"Results shape: {df.shape}")

    if not df.empty:
        print(f"Mean Pearson: {df['pearson'].mean():.4f} | Mean MSE: {df['mse'].mean():.6f}")

    if save_maps and map_store["pred_mean"]:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_maps_path)), exist_ok=True)
        arrs = {k: np.stack(map_store[k]) for k in ("pred_mean", "pred_logvar", "target")}
        meta = {k: np.array(map_store[k]) for k in ("marker", "channel_id", "dataset_name", "image_path")}
        np.savez_compressed(args.save_maps_path, **arrs, **meta)
        print(f"Saved prediction maps: {args.save_maps_path}  "
              f"(pred_mean {arrs['pred_mean'].shape}, {len(meta['marker'])} held-out predictions)")


if __name__ == "__main__":
    main()
