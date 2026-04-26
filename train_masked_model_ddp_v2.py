import os
import sys
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import matplotlib.pyplot as plt
from ruamel.yaml import YAML
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Sampler
from torchvision.transforms import Compose
from torchvision.transforms import RandomRotation, RandomCrop
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode

from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler, TestCrop
from multiplex_model.losses import beta_nll_loss
from multiplex_model.utils import ClampWithGrad, plot_reconstructs_with_uncertainty, get_scheduler_with_warmup
from multiplex_model.modules import MultiplexAutoencoder
from multiplex_model.run_utils import build_run_name_suffix, SLURM_JOB_ID

try:
    import neptune
    from neptune.utils import stringify_unsupported as _neptune_stringify
except ImportError:
    neptune = None
    _neptune_stringify = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp():
    """Initialize DDP. torchrun sets LOCAL_RANK, RANK, WORLD_SIZE automatically."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def is_main_process():
    return int(os.environ.get("RANK", 0)) == 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Distributed panel-aware batch sampler
# ---------------------------------------------------------------------------

class DistributedPanelBatchSampler(Sampler):
    """Panel-aware batch sampler sharded across DDP ranks.

    Each rank receives every ``world_size``-th batch from the full panel-grouped
    batch list. Using a shared ``seed + epoch`` guarantees all ranks see the
    same global ordering before sharding, so there is no overlap.
    """

    def __init__(self, dataset, batch_size, rank, world_size, shuffle=True, seed=42):
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Replicate the panel->indices grouping from PanelBatchSampler
        self.panel_to_indices = {}
        for idx, (_, panel_idx) in enumerate(dataset.imgs):
            if panel_idx not in self.panel_to_indices:
                self.panel_to_indices[panel_idx] = []
            self.panel_to_indices[panel_idx].append(idx)
        self.panels = list(self.panel_to_indices.keys())

    def set_epoch(self, epoch: int):
        """Call before each epoch so shuffling differs across epochs."""
        self.epoch = epoch

    def _build_all_batches(self):
        """Build the full ordered batch list deterministically from seed+epoch."""
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

        return all_batches

    def __iter__(self):
        all_batches = self._build_all_batches()
        for i, batch in enumerate(all_batches):
            if i % self.world_size == self.rank:
                yield batch

    def __len__(self):
        total_batches = sum(
            -(-len(indices) // self.batch_size)  # ceiling division
            for indices in self.panel_to_indices.values()
        )
        # Each rank gets roughly total // world_size batches
        return total_batches // self.world_size


# ---------------------------------------------------------------------------
# Logger wrappers (mirrors the original TensorboardRun interface)
# ---------------------------------------------------------------------------

def stringify_unsupported(value):
    if _neptune_stringify is None:
        return value
    return _neptune_stringify(value)


class _TensorboardSysName:
    def __init__(self, run):
        self._run = run

    def fetch(self):
        return self._run.name


class _TensorboardChannel:
    def __init__(self, run, key):
        self._run = run
        self._key = key

    def append(self, value, step=None, description=None):
        if self._key.endswith("/imgs"):
            self._run.writer.add_figure(self._key, value, global_step=step)
            if description:
                self._run.writer.add_text(
                    f"{self._key}/description",
                    description,
                    global_step=step,
                )
            return

        if isinstance(value, torch.Tensor):
            value = value.item() if value.numel() == 1 else value.mean().item()

        self._run.writer.add_scalar(self._key, value, global_step=step)


class TensorboardRun:
    def __init__(self, name, log_dir):
        if SummaryWriter is None:
            raise ImportError("TensorBoard is not installed, but logger is set to tensorboard.")
        self.name = name
        self.writer = SummaryWriter(log_dir=log_dir)
        self.writer.add_text("sys/name", name)

    def __getitem__(self, key):
        if key == "sys/name":
            return _TensorboardSysName(self)
        return _TensorboardChannel(self, key)

    def __setitem__(self, key, value):
        self.writer.add_text(key, str(value))

    def stop(self):
        self.writer.flush()
        self.writer.close()


class _NoopChannel:
    """Silently discards all logging calls on non-rank-0 processes."""
    def fetch(self):
        return "noop"

    def append(self, value, step=None, description=None):
        pass


class NoopRun:
    """Drop-in replacement for Neptune/TensorboardRun on non-rank-0 processes."""
    def __getitem__(self, key):
        return _NoopChannel()

    def __setitem__(self, key, value):
        pass

    def stop(self):
        pass


# ---------------------------------------------------------------------------
# Masking utility
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
# Training loop
# ---------------------------------------------------------------------------

def train_masked(
        model,
        optimizer,
        scheduler,
        train_dataloader,
        val_dataloader,
        device,
        run,
        marker_names_map,
        epochs=10,
        gradient_accumulation_steps=1,
        min_channels_frac=0.75,
        fully_masked_channels_max_frac=0.5,
        spatial_masking_ratio=0.6,
        mask_patch_size=8,
        early_val_epochs=0,
        early_val_checks_per_epoch=1,
        start_epoch=0,
        save_checkpoint_every=5,
        checkpoints_path='checkpoints',
    ):
    """Train a masked autoencoder with DDP support."""
    model.train()
    scaler = GradScaler()
    run_name = run['sys/name'].fetch()

    min_channels_schedule = []

    def get_min_channels_frac_for_step(step_idx: int) -> float:
        current_frac = min_channels_frac
        for boundary, value in min_channels_schedule:
            if step_idx >= boundary:
                current_frac = value
            else:
                break
        return current_frac

    if is_main_process() and not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path, exist_ok=True)
        print(f'Created checkpoints directory at {checkpoints_path}')

    steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
    early_val_epochs = 0 if early_val_epochs is None else int(early_val_epochs)
    early_val_checks_per_epoch = 1 if early_val_checks_per_epoch is None else int(early_val_checks_per_epoch)

    def _build_val_steps(steps_per_epoch: int, checks_per_epoch: int) -> set:
        if checks_per_epoch <= 1:
            return {steps_per_epoch}
        return {
            max(1, int(round(steps_per_epoch * (i + 1) / checks_per_epoch)))
            for i in range(checks_per_epoch)
        }

    early_val_steps = _build_val_steps(steps_per_epoch, early_val_checks_per_epoch)
    global_step = start_epoch * steps_per_epoch
    current_min_channels_frac = get_min_channels_frac_for_step(global_step)

    best_val_mse = float('inf')

    val_loss, val_mse = test_masked(
        model,
        val_dataloader,
        device,
        run,
        0,
        spatial_masking_ratio=spatial_masking_ratio,
        fully_masked_channels_max_frac=fully_masked_channels_max_frac,
        mask_patch_size=mask_patch_size,
        marker_names_map=marker_names_map,
    )
    if is_main_process():
        print(f'Validation loss: {val_loss:.4f}, MSE: {val_mse:.6f}')

    for epoch in range(start_epoch, epochs):
        # Reshuffle the distributed sampler deterministically for this epoch
        if hasattr(train_dataloader.batch_sampler, 'set_epoch'):
            train_dataloader.batch_sampler.set_epoch(epoch)

        model.train()
        step_in_epoch = 0
        did_epoch_end_val = False
        early_val_active = epoch < early_val_epochs and early_val_checks_per_epoch > 1

        for batch_idx, (img, channel_ids, panel_idx, img_path) in enumerate(train_dataloader):
            skip = torch.tensor(1.0 if img.shape[-1] != SIZE[0] else 0.0, device=device)
            dist.all_reduce(skip, op=dist.ReduceOp.MAX)
            if skip.item() > 0:
                if img.shape[-1] != SIZE[0]:
                    print(f'Rank {dist.get_rank()}: skipping batch {batch_idx} in epoch {epoch} due to incorrect image size: {img.shape[-1]}')
                continue

            batch_size, num_channels, H, W = img.shape

            # Channel subset sampling
            min_channels = int(np.ceil(num_channels * current_min_channels_frac))
            min_channels = max(1, min(min_channels, num_channels))

            if min_channels >= num_channels:
                num_sampled_channels = num_channels
            else:
                num_sampled_channels = np.random.randint(min_channels, num_channels)

            if num_sampled_channels < num_channels:
                new_img, new_channel_ids = [], []
                for b_i in range(batch_size):
                    idx = torch.randperm(num_channels)[:num_sampled_channels]
                    new_img.append(img[b_i:b_i+1, idx, :, :])
                    new_channel_ids.append(channel_ids[b_i:b_i+1, idx])
                img = torch.cat(new_img, dim=0)
                channel_ids = torch.cat(new_channel_ids, dim=0)

            # Full-channel masking
            max_channels_to_mask = int(np.ceil(num_sampled_channels * fully_masked_channels_max_frac))
            num_channels_to_mask = np.random.randint(1, max_channels_to_mask + 1)
            masked_img, active_channel_ids = [], []
            for b_i in range(batch_size):
                channels_to_keep = torch.randperm(num_sampled_channels)[num_channels_to_mask:]
                masked_img.append(img[b_i:b_i+1, channels_to_keep, :, :])
                active_channel_ids.append(channel_ids[b_i:b_i+1, channels_to_keep])
            masked_img = torch.cat(masked_img, dim=0)
            active_channel_ids = torch.cat(active_channel_ids, dim=0)

            masked_img = masked_img.to(device, dtype=torch.bfloat16)
            img = img.to(device, dtype=torch.bfloat16)
            masked_img = apply_patch_mask(masked_img, spatial_masking_ratio, mask_patch_size)
            channel_ids = channel_ids.to(device, dtype=torch.long)
            active_channel_ids = active_channel_ids.to(device)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                output = model(masked_img, active_channel_ids, channel_ids)['output']
                # print(f"Original output.shape: {output.shape}")
                output = output[:, :, 3:-4, 3:-4]
                mi, logvar = output.unbind(dim=-1)
                mi = torch.sigmoid(mi)
                # logvar = torch.tanh(logvar) * 5.0
                logvar = torch.clamp(logvar, min=-15.0, max=15.0)

            # Compute loss in float32 for numerical stability
            loss = beta_nll_loss(img.float(), mi.float(), logvar.float())

            loss_finite = torch.tensor(0.0 if torch.isfinite(loss) else 1.0, device=device)
            dist.all_reduce(loss_finite, op=dist.ReduceOp.MAX)
            if loss_finite.item() > 0:
                if not torch.isfinite(loss):
                    print(f'Non-finite loss at batch {batch_idx} epoch {epoch}, dataset {panel_idx}, path {img_path}')
                optimizer.zero_grad()
                continue

            scaler.scale(loss / gradient_accumulation_steps).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                step = epoch * len(train_dataloader) // gradient_accumulation_steps + batch_idx // gradient_accumulation_steps
                step_in_epoch += 1
                global_step += 1
                current_min_channels_frac = get_min_channels_frac_for_step(global_step)

                # Logging — NoopRun silently discards on non-rank-0 processes
                run['train/loss'].append(loss.item(), step=step)
                run['train/lr'].append(scheduler.get_last_lr()[0], step=step)
                run['train/µ'].append(mi.mean().item(), step=step)
                run['train/logvar'].append(logvar.mean().item(), step=step)
                run['train/mae'].append(torch.abs(img - mi).mean().item(), step=step)
                run['train/mse'].append(torch.square(img - mi).mean().item(), step=step)
                run['train/min_channels_frac'].append(current_min_channels_frac, step=step)

                if early_val_active and step_in_epoch in early_val_steps:
                    val_loss, val_mse = test_masked(
                        model,
                        val_dataloader,
                        device,
                        run,
                        epoch,
                        spatial_masking_ratio=spatial_masking_ratio,
                        fully_masked_channels_max_frac=fully_masked_channels_max_frac,
                        mask_patch_size=mask_patch_size,
                        marker_names_map=marker_names_map,
                        val_step=step,
                    )
                    if val_mse < best_val_mse:
                        best_val_mse = val_mse
                        if is_main_process():
                            raw_model = model.module if isinstance(model, DDP) else model
                            torch.save(
                                {'model_state_dict': raw_model.state_dict(), 'epoch': epoch, 'val_mse': val_mse},
                                f'{checkpoints_path}/best_model-{run_name}.pth',
                            )
                            print(f'New best model (MSE: {val_mse:.6f}) saved.')
                    if is_main_process():
                        print(f'Validation loss (epoch {epoch}, step {step_in_epoch}/{steps_per_epoch}): {val_loss:.4f}, MSE: {val_mse:.6f}')
                    model.train()
                    if step_in_epoch == steps_per_epoch:
                        did_epoch_end_val = True

        if not early_val_active or not did_epoch_end_val:
            val_loss, val_mse = test_masked(
                model,
                val_dataloader,
                device,
                run,
                epoch,
                spatial_masking_ratio=spatial_masking_ratio,
                fully_masked_channels_max_frac=fully_masked_channels_max_frac,
                mask_patch_size=mask_patch_size,
                marker_names_map=marker_names_map,
            )
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                if is_main_process():
                    raw_model = model.module if isinstance(model, DDP) else model
                    torch.save(
                        {'model_state_dict': raw_model.state_dict(), 'epoch': epoch, 'val_mse': val_mse},
                        f'{checkpoints_path}/best_model-{run_name}.pth',
                    )
                    print(f'New best model (MSE: {val_mse:.6f}) saved.')
            if is_main_process():
                print(f'Validation loss: {val_loss:.4f}, MSE: {val_mse:.6f}')

        # Checkpoint on rank 0 only
        if is_main_process() and (epoch + 1) % save_checkpoint_every == 0:
            raw_model = model.module if isinstance(model, DDP) else model
            checkpoint = {
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'min_channels_frac': current_min_channels_frac,
            }
            torch.save(checkpoint, f'{checkpoints_path}/checkpoint-{run_name}-epoch_{epoch}.pth')

        # Synchronise all ranks before moving to the next epoch
        dist.barrier()

    if is_main_process():
        raw_model = model.module if isinstance(model, DDP) else model
        final_model_path = f'{checkpoints_path}/final_model-{run_name}.pth'
        print(f'Training completed. Saving final model at {final_model_path}...')
        checkpoint = {
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epochs,
        }
        torch.save(checkpoint, final_model_path)

    dist.barrier()


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

def test_masked(
        model,
        test_dataloader,
        device,
        run,
        epoch,
        marker_names_map,
        num_plots=5,
        spatial_masking_ratio=0.6,
        fully_masked_channels_max_frac=0.5,
        mask_patch_size=8,
        val_step=None,
    ):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    running_mse = 0.0
    num_batches = 0

    # Only rank 0 plots reconstruction figures
    plot_indices = set()
    if is_main_process():
        plot_indices = set(np.random.choice(np.arange(len(test_dataloader)), size=min(num_plots, len(test_dataloader)), replace=False))

    step_base = epoch if val_step is None else int(val_step)

    with torch.no_grad():
        for idx, (img, channel_ids, panel_idx, img_path) in enumerate(test_dataloader):
            batch_size, num_channels, H, W = img.shape
            img = img.to(device, dtype=torch.float32)
            channel_ids = channel_ids.to(device, dtype=torch.long)

            max_channels_to_mask = int(np.ceil(num_channels * fully_masked_channels_max_frac))
            num_channels_to_mask = np.random.randint(1, max_channels_to_mask + 1)
            masked_img, active_channel_ids = [], []
            for b_i in range(batch_size):
                channels_to_keep = torch.randperm(num_channels)[num_channels_to_mask:]
                masked_img.append(img[b_i:b_i+1, channels_to_keep, :, :])
                active_channel_ids.append(channel_ids[b_i:b_i+1, channels_to_keep])
            masked_img = torch.cat(masked_img, dim=0)
            active_channel_ids = torch.cat(active_channel_ids, dim=0).to(device)

            masked_img = masked_img.to(device, dtype=torch.float32)
            img = img.to(device, dtype=torch.float32)
            masked_img = apply_patch_mask(masked_img, spatial_masking_ratio, mask_patch_size)

            output = model(masked_img, active_channel_ids, channel_ids)['output'][:, :, 3:-4, 3:-4]
            mi, logvar = output.unbind(dim=-1)
            mi = torch.sigmoid(mi)
            logvar = torch.clamp(logvar, min=-15.0, max=15.0)
            loss = beta_nll_loss(img, mi, logvar)

            running_loss += loss.item()
            running_mae += torch.abs(img - mi).mean().item()
            running_mse += torch.square(img - mi).mean().item()
            num_batches += 1

            if is_main_process() and idx in plot_indices:
                uncertainty_img = torch.exp(logvar / 2)
                unactive_channels = [i for i in channel_ids[0] if i not in active_channel_ids[0]]
                reconstr_img = plot_reconstructs_with_uncertainty(
                    img.float(),
                    mi.float(),
                    uncertainty_img.float(),
                    channel_ids,
                    unactive_channels,
                    markers_names_map=marker_names_map,
                    scale_by_max=True,
                )
                step = step_base * len(test_dataloader) + idx
                run['val/imgs'].append(reconstr_img, step=step)
                plt.close('all')

    # Aggregate metrics across all DDP ranks
    metrics = torch.tensor([running_loss, running_mae, running_mse, float(num_batches)], device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    total_loss, total_mae, total_mse, total_batches = metrics.tolist()

    val_loss = total_loss / total_batches
    val_mse = total_mse / total_batches
    step = step_base
    run['val/loss'].append(val_loss, step=step)
    run['val/mae'].append(total_mae / total_batches, step=step)
    run['val/mse'].append(val_mse, step=step)

    return val_loss, val_mse


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    config_path = sys.argv[1]
    yaml = YAML(typ='safe')
    with open(config_path, 'r') as f:
        config = yaml.load(f)

    # ── DDP initialisation ──────────────────────────────────────────────────
    local_rank, rank, world_size = setup_ddp()
    device = f"cuda:{local_rank}"

    if is_main_process():
        print(f'Loaded configuration from {config_path}:')
        print(config)
        print(f'World size: {world_size}')

    # ── Run name: computed on rank 0 and broadcast ──────────────────────────
    prefix = config.get("run_prefix", "").strip()
    if is_main_process():
        suffix = build_run_name_suffix()
        run_name_str = f"{prefix}_{suffix}" if prefix else suffix
    else:
        run_name_str = ""
    run_name_list = [run_name_str]
    dist.broadcast_object_list(run_name_list, src=0)
    run_name = run_name_list[0]

    # ── Logger (rank 0 only gets a real logger) ─────────────────────────────
    logger_type = config.get("logger", "neptune").lower()
    if config.get("use_tensorboard"):
        logger_type = "tensorboard"

    if is_main_process():
        if logger_type == "tensorboard":
            log_dir_root = config.get("tensorboard_log_dir", "runs")
            log_dir = os.path.join(log_dir_root, run_name)
            run = TensorboardRun(name=run_name, log_dir=log_dir)
        elif logger_type == "neptune":
            if neptune is None:
                raise ImportError("Neptune is not installed, but logger is set to neptune.")
            neptune_secrets_path = config.get("neptune_secrets_path", "./secrets/neptune.yaml")
            with open(neptune_secrets_path, 'r') as f:
                secrets = yaml.load(f)
            run = neptune.init_run(
                name=run_name,
                project=secrets['neptune_project'],
                api_token=secrets['neptune_api_token'],
                tags=config.get('tags', []),
            )
        else:
            raise ValueError(f"Unsupported logger type: {logger_type}")
    else:
        run = NoopRun()

    # ── Dataset & dataloaders ───────────────────────────────────────────────
    SIZE = config['input_image_size']
    BATCH_SIZE = config['batch_size']
    NUM_WORKERS = config['num_workers']

    PANEL_CONFIG = YAML().load(open(config['panel_config']))
    TOKENIZER = YAML().load(open(config['tokenizer_config']))
    INV_TOKENIZER = {v: k for k, v in TOKENIZER.items()}

    if is_main_process():
        print(f'Using device: {device}')
        print(f'INPUT IMAGE SIZE: {SIZE}')

    train_transform = Compose([
        RandomRotation(180, interpolation=InterpolationMode.BILINEAR),
        RandomCrop(SIZE),
    ])
    test_transform = TestCrop(SIZE[0])

    train_dataset = DatasetFromTIFF(
        panels_config=PANEL_CONFIG,
        split='train',
        marker_tokenizer=TOKENIZER,
        transform=train_transform,
        use_preprocessing=False,
        file_extension="npy",
    )
    test_dataset = DatasetFromTIFF(
        panels_config=PANEL_CONFIG,
        split='test',
        marker_tokenizer=TOKENIZER,
        transform=test_transform,
        use_preprocessing=False,
        file_extension="npy",
    )

    train_batch_sampler = DistributedPanelBatchSampler(
        train_dataset, BATCH_SIZE, rank=rank, world_size=world_size, shuffle=True,
    )
    test_batch_sampler = DistributedPanelBatchSampler(
        test_dataset, BATCH_SIZE, rank=rank, world_size=world_size, shuffle=False,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_sampler=test_batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    if is_main_process():
        print(f'Training on {len(train_dataset)} train / {len(test_dataset)} test samples')
        print(f'Batches per rank per epoch — train: {len(train_batch_sampler)}, val: {len(test_batch_sampler)}')
        print(f'Batch size (per GPU): {BATCH_SIZE}, effective batch: {BATCH_SIZE * world_size}')
        print(f'Number of workers: {NUM_WORKERS}')

    # ── Model ───────────────────────────────────────────────────────────────
    model_config = {
        'num_channels': len(TOKENIZER),
        'encoder_config': config['encoder'],
        'decoder_config': config['decoder'],
    }

    if config["model_type"] == "EquivariantConvnext":
        from multiplex_model.equivariant_modules_v2 import EquivariantMultiplexAutoencoder
        model = EquivariantMultiplexAutoencoder(**model_config).to(device)
    elif config["model_type"] == "Convnext":
        model = MultiplexAutoencoder(**model_config).to(device)
    else:
        raise ValueError(f"Unsupported model_type: {config['model_type']}")

    model = DDP(model, device_ids=[local_rank])

    if is_main_process():
        raw_model = model.module
        print(f'Model: {raw_model}')
        print(f'Trainable parameters: {sum(p.numel() for p in raw_model.parameters() if p.requires_grad):,}')

    # ── Optimiser & scheduler ───────────────────────────────────────────────
    # Scale LR linearly with world_size (linear scaling rule)
    base_lr = config['lr'] #* world_size
    final_lr = config['final_lr'] #* world_size
    weight_decay = config['weight_decay']
    gradient_accumulation_steps = config['gradient_accumulation_steps']
    epochs = config['epochs']

    total_steps = len(train_dataloader) * epochs // gradient_accumulation_steps
    if 'frac_warmup_steps' in config:
        num_warmup_steps = int(total_steps * float(config['frac_warmup_steps']))
    else:
        num_warmup_steps = int(config['num_warmup_steps'])
    num_annealing_steps = total_steps - num_warmup_steps

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scheduler = get_scheduler_with_warmup(
        optimizer, num_warmup_steps, num_annealing_steps,
        final_lr=final_lr, type='cosine', base_lr=base_lr,
    )

    # ── Checkpoint loading ──────────────────────────────────────────────────
    start_epoch = 0
    if config.get('from_checkpoint'):
        ckpt_path = config['from_checkpoint']
        if is_main_process():
            print(f'Loading model from checkpoint: {ckpt_path}')
        checkpoint = torch.load(ckpt_path, map_location=device)
        # Load into model.module (unwrapped) so all ranks sync via DDP
        model.module.load_state_dict(checkpoint['model_state_dict'])

        if config.get('resume_optimizer', False):
            if is_main_process():
                print('Resuming optimizer state from checkpoint')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            if is_main_process():
                print('Using fresh optimizer with new hyperparameters')

        if config.get('resume_scheduler', False):
            if is_main_process():
                print('Resuming scheduler state from checkpoint')
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if not config.get('resume_optimizer', False):
                scheduler.optimizer = optimizer
        else:
            if is_main_process():
                print('Using fresh scheduler with new learning rate schedule')

        start_epoch = checkpoint['epoch'] + 1
        if is_main_process():
            print(f'Resuming training from epoch {start_epoch}')

    # ── Log run metadata (rank 0 only via NoopRun on other ranks) ───────────
    run["slurm/job_id"] = SLURM_JOB_ID
    run['config'] = config

    min_channels_frac = config.get('min_channels_frac', 0.5)
    early_val_epochs = config.get('early_val_epochs', 0)
    early_val_checks_per_epoch = config.get('early_val_checks_per_epoch', 1)
    spatial_masking_ratio = config.get('spatial_masking_ratio', 0.6)
    fully_masked_channels_max_frac = config.get('fully_masked_channels_max_frac', 0.5)
    mask_patch_size = config.get('mask_patch_size', 8)
    checkpoints_path = config.get('checkpoints_dir', 'checkpoints')

    parameters = {
        "batch_size": BATCH_SIZE,
        "effective_batch_size": BATCH_SIZE * world_size,
        "world_size": world_size,
        "num_workers": NUM_WORKERS,
        "lr": base_lr,
        "final_lr": final_lr,
        "weight_decay": weight_decay,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "epochs": epochs,
        "num_warmup_steps": num_warmup_steps,
        "num_annealing_steps": num_annealing_steps,
        "model_config": stringify_unsupported(model_config),
        "min_channels_frac": min_channels_frac,
        "early_val_epochs": early_val_epochs,
        "early_val_checks_per_epoch": early_val_checks_per_epoch,
    }
    if config.get("from_checkpoint"):
        parameters["from_checkpoint"] = config["from_checkpoint"]
    run["parameters"] = parameters

    # ── Train ───────────────────────────────────────────────────────────────
    train_masked(
        model,
        optimizer,
        scheduler,
        train_dataloader,
        test_dataloader,
        device,
        epochs=epochs,
        start_epoch=start_epoch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        run=run,
        min_channels_frac=min_channels_frac,
        fully_masked_channels_max_frac=fully_masked_channels_max_frac,
        spatial_masking_ratio=spatial_masking_ratio,
        mask_patch_size=mask_patch_size,
        save_checkpoint_every=config['save_checkpoint_freq'],
        checkpoints_path=checkpoints_path,
        marker_names_map=INV_TOKENIZER,
        early_val_epochs=early_val_epochs,
        early_val_checks_per_epoch=early_val_checks_per_epoch,
    )

    run.stop()
    cleanup_ddp()
