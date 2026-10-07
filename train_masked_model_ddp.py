"""Multi-GPU (DDP) training entrypoint — the distributed twin of train_masked_model.py.

Launch with torchrun:
    torchrun --nproc_per_node=2 train_masked_model_ddp.py configs/train_equivariant_v3_config.yaml

Reuses the single-GPU model factory, masking utilities, loss and logging; adds DDP
process-group setup, a panel-aware distributed batch sampler, rank-0-only logging and
checkpointing, and cross-rank metric reduction. Works for both the vanilla and the
fully-equivariant model (autocast stays disabled for the escnn model, as single-GPU).
"""

import gc
import os
import random
import sys
from datetime import datetime

import comet_ml  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from ruamel.yaml import YAML
from torch.amp import GradScaler, autocast
from torch.nn.functional import normalize
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from torchvision.transforms import (
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
    RandomRotation,
)
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from multiplex_model.data import MultiplexDataset, TestCrop
from multiplex_model.losses import RankMe, beta_nll_loss, nll_loss
from multiplex_model.utils import (
    ClampWithGrad,
    TrainingConfig,
    apply_channel_masking,
    apply_spatial_masking,
    finish_experiment,
    get_run_name,
    get_scheduler_with_warmup,
    init_experiment,
    log_training_metrics,
    log_validation_images,
    log_validation_metrics,
    plot_reconstructs_with_masks,
)
from train_masked_model import build_model, load_model_from_checkpoint


# ── DDP helpers ────────────────────────────────────────────────────────────
def setup_ddp():
    """Initialize the process group. torchrun provides LOCAL_RANK/RANK/WORLD_SIZE."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def is_main_process():
    return int(os.environ.get("RANK", 0)) == 0


class DistributedPanelBatchSampler(Sampler):
    """Panel-aware batch sampler sharded across DDP ranks.

    Batches must stay single-panel (all images in a batch share a marker panel), so the
    stock DistributedSampler cannot be used. Every rank builds the *same* global batch
    list from a shared ``seed + epoch`` and then takes every ``world_size``-th batch.

    The list is truncated to a multiple of ``world_size`` so all ranks yield exactly the
    same number of batches — required, or DDP's gradient all-reduce desynchronizes and
    the job hangs at the end of an epoch.
    """

    def __init__(self, dataset, batch_size, rank, world_size, shuffle=True, seed=42,
                 batches_per_epoch=None):
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        # Global (pre-shard) batch count to force per epoch, or None for one pass.
        # See PanelBatchSampler for why the data-fraction ablation needs this.
        self.batches_per_epoch = batches_per_epoch

        # dataset.imgs is [(path, panel_idx), ...] — same grouping as PanelBatchSampler
        self.panel_to_indices = {}
        for idx, (_, panel_idx) in enumerate(dataset.imgs):
            self.panel_to_indices.setdefault(panel_idx, []).append(idx)
        self.panels = list(self.panel_to_indices.keys())

    def set_epoch(self, epoch: int):
        """Call before each epoch so the shuffle differs but stays identical per rank."""
        self.epoch = epoch

    def _build_all_batches(self):
        rng = random.Random(self.seed + self.epoch)
        panels = list(self.panels)
        if self.shuffle:
            rng.shuffle(panels)
        all_batches = []
        for panel in panels:
            indices = list(self.panel_to_indices[panel])
            if self.shuffle:
                rng.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                all_batches.append(indices[i : i + self.batch_size])
        if self.shuffle:
            rng.shuffle(all_batches)
        if self.batches_per_epoch is not None:
            # Repeat passes (each reshuffled) until the target, then truncate. The
            # rng is seeded from seed+epoch and every rank runs this identically,
            # so all ranks still build the SAME global list - required, or DDP's
            # all-reduce desynchronises.
            pass_idx = 0
            while len(all_batches) < self.batches_per_epoch:
                pass_idx += 1
                extra_rng = random.Random(self.seed + self.epoch + 100003 * pass_idx)
                panels_x = list(self.panels)
                if self.shuffle:
                    extra_rng.shuffle(panels_x)
                extra = []
                for panel in panels_x:
                    indices = list(self.panel_to_indices[panel])
                    if self.shuffle:
                        extra_rng.shuffle(indices)
                    for i in range(0, len(indices), self.batch_size):
                        extra.append(indices[i : i + self.batch_size])
                if self.shuffle:
                    extra_rng.shuffle(extra)
                all_batches.extend(extra)
            all_batches = all_batches[: self.batches_per_epoch]
        return all_batches

    def __iter__(self):
        all_batches = self._build_all_batches()
        usable = len(self) * self.world_size  # drop remainder -> equal batches per rank
        for i in range(self.rank, usable, self.world_size):
            yield all_batches[i]

    def __len__(self):
        if self.batches_per_epoch is not None:
            return self.batches_per_epoch // self.world_size
        total = sum(
            -(-len(v) // self.batch_size) for v in self.panel_to_indices.values()
        )
        return total // self.world_size


# ── training ───────────────────────────────────────────────────────────────
def train_masked(
    model,
    optimizer,
    scheduler,
    train_dataloader,
    val_dataloader,
    device,
    marker_names_map,
    epochs=10,
    gradient_accumulation_steps=1,
    beta=1.0,
    min_channels_frac=0.75,
    fully_masked_channels_max_frac=0.5,
    spatial_masking_ratio=0.6,
    mask_patch_size=8,
    start_epoch=0,
    save_checkpoint_every=5,
    checkpoints_path="checkpoints",
    use_amp=True,
):
    model.train()
    scaler = GradScaler(enabled=use_amp)
    run_name = get_run_name()

    if is_main_process():
        os.makedirs(checkpoints_path, exist_ok=True)

    step = start_epoch * (len(train_dataloader) // gradient_accumulation_steps)
    for epoch in range(start_epoch, epochs):
        model.train()
        if hasattr(train_dataloader.batch_sampler, "set_epoch"):
            train_dataloader.batch_sampler.set_epoch(epoch)

        for batch_idx, (img, channel_ids, panel_idx, img_path) in enumerate(
            tqdm(
                train_dataloader,
                desc=f"Epoch {epoch}",
                disable=not is_main_process(),
            )
        ):
            img = img.to(device, dtype=torch.float32)
            channel_ids = channel_ids.to(device, dtype=torch.long)

            img, channel_ids, masked_img, active_channel_ids = apply_channel_masking(
                img,
                channel_ids,
                min_channels_frac,
                fully_masked_channels_max_frac,
                apply_channel_subset_sampling=True,
            )
            masked_img, _ = apply_spatial_masking(
                masked_img, spatial_masking_ratio, mask_patch_size
            )

            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                output = model(masked_img, active_channel_ids, channel_ids)["output"]
                mi, logvar = output.unbind(dim=-1)
                mi = torch.sigmoid(mi)
                logvar = ClampWithGrad.apply(logvar, -15.0, 15.0)

            # Loss in fp32 even under bf16 autocast: the Gaussian NLL involves
            # exp(-logvar)/log terms that are precision-sensitive. Mirrors the
            # recipe the original equivariant runs used.
            loss = beta_nll_loss(img.float(), mi.float(), logvar.float(), beta=beta)

            # DDP: all ranks must make the SAME skip decision. If one rank skipped
            # a non-finite loss while others ran backward, the gradient all-reduce
            # would desynchronize and the job would hang. So reduce the flag first.
            skip = torch.tensor(
                [0.0 if torch.isfinite(loss) else 1.0], device=loss.device
            )
            dist.all_reduce(skip, op=dist.ReduceOp.MAX)
            if skip.item() > 0:
                if is_main_process():
                    print(f"Non-finite loss at batch {batch_idx}, epoch {epoch} "
                          f"- skipping on all ranks", flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss / gradient_accumulation_steps).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                if is_main_process():
                    log_training_metrics(
                        loss=loss.item(),
                        lr=scheduler.get_last_lr()[0],
                        mu=mi.mean().item(),
                        logvar=logvar.mean().item(),
                        mae=torch.abs(img - mi).mean().item(),
                        mse=torch.square(img - mi).mean().item(),
                        step=step,
                    )
                step += 1


        # --- peak GPU memory for this epoch's training pass (OOM headroom) ---
        if is_main_process():
            _pk = torch.cuda.max_memory_allocated() / 1024**3
            _rs = torch.cuda.max_memory_reserved() / 1024**3
            print(f"[mem] epoch {epoch}: peak allocated {_pk:.2f} GiB | "
                  f"peak reserved {_rs:.2f} GiB", flush=True)
        torch.cuda.reset_peak_memory_stats()

        test_masked(
            model,
            val_dataloader,
            device,
            epoch,
            spatial_masking_ratio=spatial_masking_ratio,
            fully_masked_channels_max_frac=fully_masked_channels_max_frac,
            mask_patch_size=mask_patch_size,
            marker_names_map=marker_names_map,
        )

        if is_main_process():
            raw_model = model.module
            checkpoint = {
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
            }
            if hasattr(raw_model, "get_architecture_config"):
                checkpoint["model_config"] = raw_model.get_architecture_config()
            if (epoch + 1) % save_checkpoint_every == 0:
                torch.save(
                    checkpoint,
                    f"{checkpoints_path}/checkpoint-{run_name}-epoch_{epoch}.pth",
                )
            torch.save(checkpoint, f"{checkpoints_path}/last_checkpoint-{run_name}.pth")

        # keep ranks together: rank 0 may spend minutes writing a 10 GB checkpoint
        dist.barrier()

    if is_main_process():
        raw_model = model.module
        final_model_path = f"{checkpoints_path}/final_model-{run_name}.pth"
        print(f"Training completed. Saving final model at {final_model_path}...")
        checkpoint = {"model_state_dict": raw_model.state_dict()}
        if hasattr(raw_model, "get_architecture_config"):
            checkpoint["model_config"] = raw_model.get_architecture_config()
        torch.save(checkpoint, final_model_path)


def test_masked(
    model,
    test_dataloader,
    device,
    epoch,
    marker_names_map,
    num_plots=4,
    spatial_masking_ratio=0.6,
    fully_masked_channels_max_frac=0.5,
    mask_patch_size=8,
):
    # unwrap DDP: the split encode()/decode() API lives on the underlying model
    raw_model = model.module
    raw_model.eval()

    running_loss = running_mae = running_mse = 0.0
    num_batches = 0

    plot_indices = set()
    if is_main_process():
        plot_indices = set(
            np.random.choice(
                np.arange(len(test_dataloader)),
                size=min(num_plots, len(test_dataloader)),
                replace=False,
            )
        )

    all_latents, all_channel_variances, all_channel_maes = [], [], []

    with torch.no_grad():
        for idx, (img, channel_ids, panel_idx, img_path) in enumerate(
            tqdm(
                test_dataloader,
                desc=f"Testing epoch {epoch}",
                disable=not is_main_process(),
            )
        ):
            img = img.to(device, dtype=torch.float32)
            channel_ids = channel_ids.to(device, dtype=torch.long)

            _, _, masked_img, active_channel_ids = apply_channel_masking(
                img,
                channel_ids,
                fully_masked_channels_max_frac=fully_masked_channels_max_frac,
                apply_channel_subset_sampling=False,
            )
            masked_img, pixel_mask = apply_spatial_masking(
                masked_img, spatial_masking_ratio, mask_patch_size
            )

            latent = raw_model.encode(masked_img, active_channel_ids)["output"]
            output = raw_model.decode(latent, channel_ids)
            mi, logvar = output.unbind(dim=-1)
            mi = torch.sigmoid(mi)

            latent = normalize(latent.mean(dim=(2, 3)), p=2, dim=1)
            all_latents.append(latent.cpu())
            all_channel_variances.append(torch.exp(logvar).mean(dim=(0, 2, 3)).cpu())
            all_channel_maes.append(torch.abs(img - mi).mean(dim=(0, 2, 3)).cpu())

            loss = nll_loss(img, mi, logvar)
            running_loss += loss.item()
            running_mae += torch.abs(img - mi).mean().item()
            running_mse += torch.square(img - mi).mean().item()
            num_batches += 1

            if is_main_process() and idx in plot_indices:
                unactive_channels = [
                    i for i in channel_ids[0] if i not in active_channel_ids[0]
                ]
                masked_channels_names = " | ".join(
                    marker_names_map[i.item()] for i in unactive_channels
                )
                reconstr_img = plot_reconstructs_with_masks(
                    img,
                    mi,
                    pixel_mask,
                    channel_ids,
                    unactive_channels,
                    markers_names_map=marker_names_map,
                    ncols=9,
                )
                log_validation_images(
                    fig=reconstr_img,
                    panel_idx=panel_idx[0],
                    img_path=img_path[0],
                    epoch=epoch,
                    masked_channels_names=masked_channels_names,
                    img_idx=idx,
                )
                plt.close("all")

    # reduce scalar metrics over all ranks so the logged value covers the whole val set
    metrics = torch.tensor(
        [running_loss, running_mae, running_mse, float(num_batches)], device=device
    )
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    total_loss, total_mae, total_mse, total_batches = metrics.tolist()
    val_loss = total_loss / total_batches
    val_mae = total_mae / total_batches
    val_mse = total_mse / total_batches

    val_metrics = {"val_loss": val_loss, "val_mae": val_mae, "val_mse": val_mse, "epoch": epoch}

    if is_main_process():
        # latent diagnostics are computed on rank 0's shard only (monitoring signal,
        # not a headline metric) — gathering full latents across ranks is not worth it.
        latents = torch.cat(all_latents)
        rankme = RankMe(latents)
        acv = torch.cat(all_channel_variances)
        acm = torch.cat(all_channel_maes)
        variance_mae_corr = torch.corrcoef(
            torch.stack([acv.flatten(), acm.flatten()])
        )[0, 1].item()

        log_validation_metrics(
            val_loss=val_loss,
            val_mae=val_mae,
            val_mse=val_mse,
            latent_rankme=rankme,
            variance_mae_correlation=variance_mae_corr,
            epoch=epoch,
        )
        print(f"{'=' * 40} EPOCH {epoch + 1} {'=' * 40}")
        print(f"NLL: {val_loss:.4f}")
        print(f"MAE: {val_mae:.6f}")
        print(f"MSE: {val_mse:.6f}")
        print(f"Pearson MAE vs Var: {variance_mae_corr:.4f}  (rank-0 shard)")
        print("=" * 90)
        print()

    return val_metrics


# ── entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    local_rank, rank, world_size = setup_ddp()
    device = f"cuda:{local_rank}"

    config_path = sys.argv[1]
    yaml = YAML(typ="safe")
    with open(config_path, "r") as f:
        raw_config = yaml.load(f)
    config = TrainingConfig(**raw_config)

    if is_main_process():
        print(f"World size: {world_size} | device: {device} | model_type: {config.model_type}")

    SIZE = config.input_image_size
    BATCH_SIZE = config.batch_size
    NUM_WORKERS = config.num_workers
    TOKENIZER = config.tokenizer_config
    INV_TOKENIZER = {v: k for k, v in TOKENIZER.items()}

    # train_augmentation: 'rotflip' (default, the ImmuVis protocol) applies
    # RandomRotation(180) + RandomHorizontalFlip, i.e. the full O(2) symmetry the
    # equivariant model encodes architecturally. 'crop_only' drops both, leaving
    # the RandomCrop the pipeline needs to reach the target size.
    _aug = config.train_augmentation
    print(f"Train augmentation: {_aug}")
    _pre = ([RandomRotation(180, interpolation=InterpolationMode.BILINEAR)]
            if _aug == "rotflip" else [])
    _post = [RandomHorizontalFlip()] if _aug == "rotflip" else []
    train_transform = Compose(
        [
            *_pre,
            RandomCrop(SIZE),
            *_post,
        ]
    )
    test_transform = TestCrop(SIZE[0])
    dataset_kwargs = config.data_config.model_dump()

    # Train-only subsetting; validation stays full size (see train_masked_model.py).
    train_dataset = MultiplexDataset(
        panels_config=config.panel_config,
        split="train",
        marker_tokenizer=TOKENIZER,
        transform=train_transform,
        subset_fraction=config.train_subset_fraction,
        subset_seed=config.train_subset_seed,
        **dataset_kwargs,
    )
    test_dataset = MultiplexDataset(
        panels_config=config.panel_config,
        split="test",
        marker_tokenizer=TOKENIZER,
        transform=test_transform,
        **dataset_kwargs,
    )

    train_batch_sampler = DistributedPanelBatchSampler(
        train_dataset, BATCH_SIZE, rank, world_size, shuffle=True,
        batches_per_epoch=config.train_batches_per_epoch,
    )
    test_batch_sampler = DistributedPanelBatchSampler(
        test_dataset, BATCH_SIZE, rank, world_size, shuffle=False
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_sampler=test_batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    if is_main_process():
        print(
            f"Training on {len(train_dataset)} train / {len(test_dataset)} test samples\n"
            f"Batches per rank — train: {len(train_batch_sampler)}, val: {len(test_batch_sampler)}\n"
            f"Batch size per GPU: {BATCH_SIZE}, effective batch: {BATCH_SIZE * world_size}"
        )

    num_channels = len(TOKENIZER)
    model_config = {
        "num_channels": num_channels,
        "encoder_config": config.encoder_config.model_dump(),
        "decoder_config": config.decoder_config.model_dump(),
    }

    start_epoch = 0
    checkpoint = None
    if config.resolve_checkpoint():
        # Resolve to an absolute path so the log is unambiguous about WHICH file was
        # read (from_checkpoint: last is relative to checkpoints_dir + run_name).
        ckpt_path = os.path.abspath(config.from_checkpoint)
        if is_main_process():
            _stat = os.stat(ckpt_path)
            print(
                f"\n{'=' * 90}\n"
                f"RESUMING FROM CHECKPOINT\n"
                f"  path      : {ckpt_path}\n"
                f"  size      : {_stat.st_size / 1024**3:.2f} GiB\n"
                f"  modified  : {datetime.fromtimestamp(_stat.st_mtime):%Y-%m-%d %H:%M:%S}\n",
                flush=True,
            )
        # Load to CPU, not `device`: the state dicts are copied into the model and
        # the optimizer anyway (optimizer.load_state_dict casts state to each
        # param's device), so staging the whole ~10 GiB file in VRAM only burns
        # headroom. It is freed below once both restores are done.
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model = load_model_from_checkpoint(
            checkpoint, model_config, config.model_type
        ).to(device)
        start_epoch = checkpoint.get("epoch", -1) + 1
        if is_main_process():
            print(
                f"  ckpt epoch: {checkpoint.get('epoch')} (0-indexed)\n"
                f"  resuming at epoch {start_epoch} -> {config.epochs - 1} "
                f"({config.epochs - start_epoch} epochs left)\n"
                f"  keys      : {sorted(checkpoint.keys())}\n"
                f"{'=' * 90}\n",
                flush=True,
            )
    else:
        # Loud, because silently restarting a long run from scratch wastes days.
        if is_main_process():
            print(
                f"\n{'=' * 90}\n"
                f"NO CHECKPOINT LOADED - TRAINING FROM SCRATCH (epoch 0)\n"
                f"  from_checkpoint resolved to: {config.from_checkpoint!r}\n"
                f"  checkpoints_dir : {os.path.abspath(config.checkpoints_dir)}\n"
                f"  run_name        : {config.run_name}\n"
                f"{'=' * 90}\n",
                flush=True,
            )
        model = build_model(model_config, config.model_type).to(device)

    model = DDP(model, device_ids=[local_rank])

    total_steps = (
        len(train_dataloader) * config.epochs // config.gradient_accumulation_steps
    )
    num_warmup_steps = int(total_steps * config.frac_warmup_steps)
    optimizer = optim.AdamW(
        model.parameters(), lr=config.peak_lr, weight_decay=config.weight_decay
    )
    scheduler = get_scheduler_with_warmup(
        optimizer,
        num_warmup_steps,
        total_steps - num_warmup_steps,
        final_lr=config.final_lr,
        peak_lr=config.peak_lr,
        type="cosine",
    )
    if checkpoint is not None:
        if is_main_process():
            print(
                f"[resume] fresh scheduler: total_steps={total_steps} "
                f"warmup={num_warmup_steps} | lr before restore={scheduler.get_last_lr()[0]:.6e}",
                flush=True,
            )
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if is_main_process():
                print("[resume] optimizer state RESTORED", flush=True)
        elif is_main_process():
            print("[resume] WARNING: no optimizer_state_dict in checkpoint", flush=True)
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if is_main_process():
                print(
                    f"[resume] scheduler state RESTORED: step {scheduler.last_epoch} "
                    f"| lr now {scheduler.get_last_lr()[0]:.6e} "
                    f"({'warmup' if scheduler.last_epoch < num_warmup_steps else 'cosine anneal'})",
                    flush=True,
                )
        elif is_main_process():
            print("[resume] WARNING: no scheduler_state_dict in checkpoint", flush=True)

        # Drop the checkpoint now that model/optimizer/scheduler all hold their own
        # copies. Keeping it alive pins a redundant copy of the optimizer state for
        # the whole run (~2.3 GiB here), which is what OOM'd job 3063595 at epoch 29:
        # the fresh run peaked at a flat 32.12 GiB, the resumed one at 34.29 GiB.
        del checkpoint
        checkpoint = None
        gc.collect()
        torch.cuda.empty_cache()
        if is_main_process():
            print(
                f"[resume] checkpoint released | "
                f"allocated {torch.cuda.memory_allocated() / 1024**3:.2f} GiB, "
                f"reserved {torch.cuda.memory_reserved() / 1024**3:.2f} GiB",
                flush=True,
            )

    if is_main_process():
        init_experiment(config.model_dump())

    train_masked(
        model,
        optimizer,
        scheduler,
        train_dataloader,
        test_dataloader,
        device,
        marker_names_map=INV_TOKENIZER,
        epochs=config.epochs,
        start_epoch=start_epoch,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        min_channels_frac=config.min_channels_frac,
        spatial_masking_ratio=config.spatial_masking_ratio,
        fully_masked_channels_max_frac=config.fully_masked_channels_max_frac,
        mask_patch_size=config.mask_patch_size,
        save_checkpoint_every=config.save_checkpoint_freq,
        checkpoints_path=config.checkpoints_dir,
        beta=config.beta,
        # bf16 for the equivariant model too: the original pipeline trained
        # FullyEquivariantConvnextV3 under bf16 autocast, and those checkpoints
        # retain EXACT 90-degree equivariance (corr 1.0000) — the steerable
        # structure survives reduced-precision training. Loss stays fp32.
        use_amp=True,
    )

    if is_main_process():
        finish_experiment()
    dist.destroy_process_group()
