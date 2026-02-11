"""
Inference script for extracting latent embeddings from a trained multiplex autoencoder.

For each image, runs the encoder in 8 orientations (4 rotations x 2 flips)
and stores the latent embedding, reconstruction loss, MAE, and rotation error.
Each image is processed as a whole (no patch grid).

Usage:
    python infer_embeddings.py <config.yaml>
"""

import os
import sys
import pickle
import numpy as np
import torch
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

import torch.nn.functional as F

from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler, TestCrop
from multiplex_model.losses import nll_loss
from multiplex_model.modules import MultiplexAutoencoder


def _pad_collate(batch):
    """Custom collate that pads images to the max spatial size in the batch."""
    imgs, channel_ids_list, panel_idxs, img_paths = zip(*batch)
    max_h = max(img.shape[-2] for img in imgs)
    max_w = max(img.shape[-1] for img in imgs)
    orig_sizes = [(img.shape[-2], img.shape[-1]) for img in imgs]
    padded = []
    for img in imgs:
        pad_h = max_h - img.shape[-2]
        pad_w = max_w - img.shape[-1]
        padded.append(F.pad(img, (0, pad_w, 0, pad_h), value=0.0))
    return (
        torch.stack(padded),
        torch.stack(channel_ids_list),
        panel_idxs,
        img_paths,
        orig_sizes,
    )


ORIENTATIONS = [(r, f) for r in (0, 90, 180, 270) for f in (0, 1)]


def infer_split(model, dataloader, device, output_crop):
    """Run inference on a dataloader, extracting embeddings in 8 orientations per image.

    All 8 orientations are stacked into a single batched forward pass (B*8)
    for maximum GPU utilisation.
    """
    model.eval()
    results_images = {}

    with torch.no_grad():
        for idx, (img, channel_ids, panel_idx, img_path, orig_sizes) in enumerate(tqdm(dataloader)):
            img = img.to(device, dtype=torch.float32)
            channel_ids = channel_ids.to(device, dtype=torch.long)
            B = img.shape[0]

            # Build all 8 orientations for every image in the batch: [B*8, C, H', W']
            transformed_list = []
            for rotation, flip in ORIENTATIONS:
                t = img
                if rotation > 0:
                    t = torch.rot90(t, k=rotation // 90, dims=[-2, -1])
                if flip > 0:
                    t = torch.flip(t, dims=[-1])
                transformed_list.append(t)
            all_transformed = torch.cat(transformed_list, dim=0)  # [B*8, C, H', W']

            # Expand channel_ids to match: repeat each row 8 times
            all_channel_ids = channel_ids.repeat(8, 1)  # [B*8, num_markers]

            # Single forward pass for all orientations
            outputs = model(all_transformed, all_channel_ids, all_channel_ids, True)
            all_embedding = outputs["features"][-1]  # [B*8, D, eH, eW]

            reconstruction = outputs["output"]
            if output_crop == "3:-4":
                reconstruction = reconstruction[:, :, 3:-4, 3:-4, :]
            else:
                reconstruction = reconstruction[:, :, 1:, 1:, :]
            all_mi, all_logsigma = reconstruction.unbind(dim=-1)
            all_mi = torch.sigmoid(all_mi)
            all_logsigma = torch.tanh(all_logsigma) * 5.0

            # Split results back: orientation o, image b -> index o*B + b
            batch_results = [{} for _ in range(B)]
            for o, (rotation, flip) in enumerate(ORIENTATIONS):
                key = f"r{rotation}_{'f' if flip > 0 else 'nf'}"
                for b in range(B):
                    i = o * B + b  # index into the B*8 batch
                    embedding = all_embedding[i:i+1]
                    mi = all_mi[i:i+1]
                    logsigma = all_logsigma[i:i+1]
                    transformed = all_transformed[i:i+1]

                    # Unpad: use only the original image region for metrics
                    oH, oW = orig_sizes[b]
                    if rotation in (90, 270):
                        oH, oW = oW, oH
                    _, _, rH, rW = mi.shape
                    minH, minW = min(rH, oH), min(rW, oW)
                    target_b = transformed[:, :, :minH, :minW]
                    mi_b = mi[:, :, :minH, :minW]
                    logsigma_b = logsigma[:, :, :minH, :minW]

                    loss_b = nll_loss(target_b, mi_b, logsigma_b)
                    mae_b = torch.abs(target_b - mi_b).mean()

                    # Undo transformations on embedding to align back
                    embedding_aligned = embedding.clone()
                    if flip > 0:
                        embedding_aligned = torch.flip(embedding_aligned, dims=[-1])
                    if rotation > 0:
                        embedding_aligned = torch.rot90(embedding_aligned, k=(-rotation // 90), dims=[-2, -1])

                    rotation_error = 0.0
                    if rotation > 0 or flip > 0:
                        org_emb = torch.tensor(
                            batch_results[b]["r0_nf"]["embedding"]
                        ).to(device)
                        rotation_error = (
                            (embedding_aligned - org_emb) ** 2
                        ).mean() / (org_emb ** 2).mean()
                        rotation_error = rotation_error.item()

                    batch_results[b][key] = {
                        "embedding": embedding.detach().cpu().numpy(),
                        "loss": loss_b.item(),
                        "mae": mae_b.item(),
                        "rotation_error": rotation_error,
                    }

            for b in range(B):
                results_images[img_path[b]] = batch_results[b]

    return results_images


if __name__ == "__main__":
    config_path = sys.argv[1]
    yaml = YAML(typ="safe")
    with open(config_path, "r") as f:
        config = yaml.load(f)

    print(f"Loaded configuration from {config_path}:")
    print(config)

    device = config["device"]
    print(f"Using device: {device}")

    PANEL_CONFIG = YAML().load(open(config["panel_config"]))
    TOKENIZER = YAML().load(open(config["tokenizer_config"]))
    INV_TOKENIZER = {v: k for k, v in TOKENIZER.items()}

    # Resolve which panel's marker indices to use for encoding
    infer_panel = config.get("infer_panel", PANEL_CONFIG["datasets"][0])
    panel_markers = PANEL_CONFIG["markers"][infer_panel]
    panel_indices = [TOKENIZER[m] for m in panel_markers]
    print(f"Inference panel: {infer_panel}, {len(panel_indices)} markers")

    BATCH_SIZE = config.get("batch_size", 8)

    # Build model
    model_config = {
        "num_channels": len(TOKENIZER),
        "encoder_config": config["encoder"],
        "decoder_config": config["decoder"],
    }

    if config["model_type"] == "EquivariantConvnext":
        from multiplex_model.equivariant_modules import EquivariantMultiplexAutoencoder
        model = EquivariantMultiplexAutoencoder(**model_config).to(device)
    elif config["model_type"] == "Convnext":
        model = MultiplexAutoencoder(**model_config).to(device)
    else:
        raise ValueError(f"Unknown model_type: {config['model_type']}")

    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

    # Load checkpoint
    ckpt_path = config["from_checkpoint"]
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ckpt_name = os.path.basename(ckpt_path).replace("checkpoint-", "").replace(".pth", "")

    # Output crop style
    output_crop = config.get("output_crop", "1:")

    # Determine which splits to run
    splits = config.get("splits", ["train", "test"])
    output_dir = config.get("output_dir", f"embeddings/{ckpt_name}")
    os.makedirs(output_dir, exist_ok=True)

    for split in splits:
        print(f"\n--- Processing split: {split} ---")
        dataset = DatasetFromTIFF(
            panels_config=PANEL_CONFIG,
            split=split,
            marker_tokenizer=TOKENIZER,
            transform=Compose([]),
            use_preprocessing=False,
            file_extension="npy",
        )
        batch_sampler = PanelBatchSampler(dataset, BATCH_SIZE, shuffle=False)
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=config.get("num_workers", 4), collate_fn=_pad_collate)
        print(f"  {len(dataset)} images, batch_size={BATCH_SIZE}")

        results = infer_split(
            model, dataloader, device, output_crop,
        )

        safe_panel_name = infer_panel.replace("-", "_")
        out_path = os.path.join(output_dir, f"{safe_panel_name}_{split}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(results, f)
        print(f"  Saved {len(results)} images to {out_path}")

    print("\nDone.")
