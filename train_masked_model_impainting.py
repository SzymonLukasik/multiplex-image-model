import os
import sys
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import neptune
from neptune.utils import stringify_unsupported
from ruamel.yaml import YAML
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.transforms import RandomRotation, RandomCrop
from tqdm import tqdm
import torch.nn.functional as F


from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler, TestCrop
from multiplex_model.losses import nll_loss
from multiplex_model.utils import ClampWithGrad, plot_reconstructs_with_uncertainty, get_scheduler_with_warmup
from multiplex_model.modules import MultiplexAutoencoder
from multiplex_model.run_utils import build_run_name_suffix, SLURM_JOB_ID

def apply_patch_mask(x: torch.Tensor, ratio: float, patch_size: int) -> torch.Tensor:
    # x: [B, C, H, W]
    B, C, H, W = x.shape

    pad_h = (patch_size - (H % patch_size)) % patch_size
    pad_w = (patch_size - (W % patch_size)) % patch_size

    # pad order in F.pad for 4D is (left, right, top, bottom)
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)

    Hp, Wp = x.shape[-2:]
    h = Hp // patch_size
    w = Wp // patch_size
    total_patches = h * w

    # [B, C, h, w, ps, ps]
    patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).contiguous()
    # [B, C, total_patches, ps*ps]
    patches = patches.view(B, C, total_patches, patch_size * patch_size)

    mask = (torch.rand((B, C, total_patches), device=x.device) < ratio)
    patches[mask] = 0.0

    # fold back
    x = patches.view(B, C, h, w, patch_size, patch_size)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, Hp, Wp)

    # unpad back to original size
    return x[..., :H, :W]

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
        start_epoch=0,
        save_checkpoint_every=5,
        checkpoints_path='checkpoints'
    ):
    """Train a masked autoencoder (decode the remaining channels) with the given parameters."""
    model.train()
    scaler = GradScaler()
    run_name = run['sys/name'].fetch()

    min_channels_schedule = []
    # if min_channels_frac_dict:
    #     min_channels_schedule = sorted(
    #         (int(step_idx), float(value))
    #         for step_idx, value in min_channels_frac_dict.items()
    #     )

    def get_min_channels_frac_for_step(step_idx: int) -> float:
        current_frac = min_channels_frac
        for boundary, value in min_channels_schedule:
            if step_idx >= boundary:
                current_frac = value
            else:
                break
        return current_frac

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path, exist_ok=True)
        print(f'Created checkpoints directory at {checkpoints_path}')

    steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
    global_step = start_epoch * steps_per_epoch
    current_min_channels_frac = get_min_channels_frac_for_step(global_step)
    val_loss = test_masked(
        model, 
        val_dataloader, 
        device, 
        run, 
        None, 
        spatial_masking_ratio=spatial_masking_ratio,
        fully_masked_channels_max_frac=fully_masked_channels_max_frac,
        mask_patch_size=mask_patch_size,
        marker_names_map=marker_names_map,
    )
    print(f'Validation loss: {val_loss:.4f}')

    for epoch in range(start_epoch, epochs):
        model.train()
        for batch_idx, (img, channel_ids, panel_idx, img_path) in enumerate(train_dataloader):
            if img.shape[-1] != SIZE[0]:
                print(f'Skipping batch {batch_idx} in epoch {epoch} due to incorrect image size: {img.shape[-1]}')
                continue
            # print(f'Processing batch {batch_idx} in epoch {epoch}...')
            # print(f'Batch size: {img.shape[0]}, Image shape: {img.shape}, Channel IDs shape: {channel_ids.shape}, Panel index: {panel_idx}, Image path: {img_path}')
            batch_size, num_channels, H, W = img.shape
            # trim channel_ids for debugging to 10 first channels
            # channel_ids = channel_ids[:, :10]

            # Randomly sample a subset of channels to keep
            min_channels = int(np.ceil(num_channels * current_min_channels_frac))
            min_channels = max(1, min(min_channels, num_channels))

            if min_channels >= num_channels:
                num_sampled_channels = num_channels
            else:
                num_sampled_channels = np.random.randint(min_channels, num_channels)
            if num_sampled_channels < num_channels:
                new_img = []
                new_channel_ids = []
                for b_i in range(batch_size):
                    channels_subset_idx = torch.randperm(num_channels)[:num_sampled_channels]
                    new_img.append(img[b_i:b_i+1, channels_subset_idx, :, :])
                    new_channel_ids.append(channel_ids[b_i:b_i+1, channels_subset_idx])
                img = torch.cat(new_img, dim=0)
                channel_ids = torch.cat(new_channel_ids, dim=0)


            # sample full channels to mask (drop)
            max_channels_to_mask = int(np.ceil(num_sampled_channels * fully_masked_channels_max_frac))
            num_channels_to_mask = np.random.randint(1, max_channels_to_mask + 1)
            masked_img = []
            active_channel_ids = []
            for b_i in range(batch_size):
                channels_to_keep = torch.randperm(num_sampled_channels)[num_channels_to_mask:]
                masked_img.append(img[b_i:b_i+1, channels_to_keep, :, :])
                active_channel_ids.append(channel_ids[b_i:b_i+1, channels_to_keep])
            masked_img = torch.cat(masked_img, dim=0) # [B, C_new, H, W]
            active_channel_ids = torch.cat(active_channel_ids, dim=0) # [B, C_new]
            num_active_channels = masked_img.shape[1]

            masked_img = masked_img.to(device, dtype=torch.bfloat16)
            img = img.to(device, dtype=torch.bfloat16)
            masked_img = apply_patch_mask(masked_img, spatial_masking_ratio, mask_patch_size)
            channel_ids = channel_ids.to(device, dtype=torch.long)
            active_channel_ids = active_channel_ids.to(device)
                 

            # we use float16
            with autocast(device_type='cuda', dtype=torch.bfloat16):#, dtype=torch.bfloat16):
                # output = model(masked_img, active_channel_ids, channel_ids)['output']
                output = model(masked_img, active_channel_ids, channel_ids)['output'][:, :, 3:-4, 3:-4]
                # output = model(masked_img, active_channel_ids, active_channel_ids)['output'][:, :, 3:-4, 3:-4]
                # print(f"output shape: {output.shape}")
                mi, logsigma = output.unbind(dim=-1)
                mi = torch.sigmoid(mi)
                # print(f'Mean of mi: {mi.mean().item()}, Mean of logsigma: {logsigma.mean().item()}')

                # Apply ClampWithGrad to logsigma for stability
                # logsigma = ClampWithGrad.apply(logsigma, -15.0, 15.0)
                logsigma = torch.tanh(logsigma) * 5.0  # Scale logsigma to a reasonable range
                loss = nll_loss(img, mi, logsigma)
                # print(loss.item())

                # sanity check if loss is finite
                if not torch.isfinite(loss):
                    print(f'Non-finite loss encountered at batch {batch_idx} in epoch {epoch}. Skipping batch.')
                    print(f'Dataset: {panel_idx}')
                    print(f'Image path: {img_path}')
                    print(f'Mi: {mi}, Logsigma: {logsigma}')
                    assert False, "Non-finite loss encountered. Check the model and data."

            scaler.scale(loss / gradient_accumulation_steps).backward()

            if (batch_idx+1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                step = epoch * len(train_dataloader) // gradient_accumulation_steps + batch_idx // gradient_accumulation_steps
                run['train/loss'].append(loss.item(), step=step)
                run['train/lr'].append(scheduler.get_last_lr()[0], step=step)
                run['train/µ'].append(mi.mean().item(), step=step)
                run['train/logvar'].append(logsigma.mean().item(), step=step)
                run['train/mae'].append(torch.abs(img - mi).mean().item(), step=step)
                run['train/mse'].append(torch.square(img - mi).mean().item())
                run['train/min_channels_frac'].append(current_min_channels_frac, step=step)
                global_step += 1
                current_min_channels_frac = get_min_channels_frac_for_step(global_step)
        # scheduler.step()

        val_loss = test_masked(
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
        print(f'Validation loss: {val_loss:.4f}')

        if (epoch + 1) % save_checkpoint_every == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'min_channels_frac': current_min_channels_frac,
            }
            torch.save(checkpoint, f'{checkpoints_path}/checkpoint-{run_name}-epoch_{epoch}.pth')

    final_model_path = f'{checkpoints_path}/final_model-{run_name}.pth'
    print(f'Training completed. Saving final model at {final_model_path}...')
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epochs,
    }
    torch.save(checkpoint, final_model_path)


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
        ):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    running_mse = 0.0
    plot_indices = np.random.choice(np.arange(len(test_dataloader)), size=num_plots, replace=False)
    plot_indices = set(plot_indices)
    rand_gen = torch.Generator().manual_seed(42)
    with torch.no_grad():
        for idx, (img, channel_ids, panel_idx, img_path) in enumerate(test_dataloader):
            batch_size, num_channels, H, W = img.shape
            img = img.to(device, dtype=torch.bfloat16)
            channel_ids = channel_ids.to(device, dtype=torch.long)

            # sample full channels to mask (drop)
            max_channels_to_mask = int(np.ceil(num_channels * fully_masked_channels_max_frac))
            num_channels_to_mask = np.random.randint(1, max_channels_to_mask + 1)
            masked_img = []
            active_channel_ids = []
            for b_i in range(batch_size):
                channels_to_keep = torch.randperm(num_channels)[num_channels_to_mask:]
                masked_img.append(img[b_i:b_i+1, channels_to_keep, :, :])
                active_channel_ids.append(channel_ids[b_i:b_i+1, channels_to_keep])
            masked_img = torch.cat(masked_img, dim=0) # [B, C_new, H, W]
            active_channel_ids = torch.cat(active_channel_ids, dim=0).to(device) # [B, C_new]
            num_active_channels = masked_img.shape[1]

            # randomly mask spatial_masking_ratio image patches
            # unfold image into patches
            
            masked_img = masked_img.to(device, dtype=torch.float32)
            img = img.to(device, dtype=torch.float32)
            masked_img = apply_patch_mask(masked_img, spatial_masking_ratio, mask_patch_size)

            # output = model(masked_img, active_channel_ids, channel_ids)['output']
            # with autocast(device_type='cuda', dtype=torch.bfloat16):#, dtype=torch.bfloat16):
            output = model(masked_img, active_channel_ids, channel_ids)['output'][:, :, 3:-4, 3:-4]  # Remove padding
            # output = model(masked_img, active_channel_ids, active_channel_ids)['output'][:, :, 3:-4, 3:-4]  # Remove padding
            mi, logsigma = output.unbind(dim=-1)
            mi = torch.sigmoid(mi)

            logsigma = torch.tanh(logsigma) * 5.0  # Scale logsigma to a reasonable range
            loss = nll_loss(img, mi, logsigma)
            running_loss += loss.item()
            running_mae += torch.abs(img - mi).mean().item()
            running_mse += torch.square(img - mi).mean().item()

            if idx in plot_indices:
                uncertainty_img = torch.exp(logsigma)
                unactive_channels = [i for i in channel_ids[0] if i not in active_channel_ids[0]]
                # unactive_channels = []
                masked_channels_names = '\n'.join([marker_names_map[i.item()] for i in unactive_channels])

                reconstr_img = plot_reconstructs_with_uncertainty(
                    img,
                    mi,
                    uncertainty_img,
                    channel_ids,
                    unactive_channels,
                    markers_names_map=marker_names_map,
                    scale_by_max=True
                )
                run['val/imgs'].append(
                    reconstr_img, 
                    description=f'Resuilting outputs (variance scaled by min-max)  (dataset {panel_idx[0]}, image {img_path[0]}, epoch {epoch})'
                                '\n\nMasked channels: {}'.format(masked_channels_names)
                )
                plt.close('all')
                
    step = None #(epoch + 1) * len(test_dataloader)
    val_loss = running_loss / len(test_dataloader)
    run['val/loss'].append(val_loss, step=step)
    run['val/mae'].append(running_mae / len(test_dataloader), step=step)
    run['val/mse'].append(running_mse / len(test_dataloader), step=step)

    return val_loss

if __name__ == '__main__':
    # Load the configuration file
    config_path = sys.argv[1]
    yaml = YAML(typ='safe')
    with open(config_path, 'r') as f:
        config = yaml.load(f)
    
    with open("/home/szlukasik/immu-vis/multiplex-image-model/secrets/neptune.yaml", 'r') as f:
        secrets = yaml.load(f)

    prefix = config.get("run_prefix", "").strip()         # empty by default
    suffix = build_run_name_suffix()                               # always unique
    run_name = f"{prefix}_{suffix}" if prefix else suffix

    run = neptune.init_run(
        name=run_name,
        project=secrets['neptune_project'],
        api_token=secrets['neptune_api_token'],
        tags=config['tags'],
    )

    device = config['device']
    print(f'Using device: {device}')

    SIZE = config['input_image_size']
    print(f"INPUT IMAGE SIZE: {SIZE}")
    BATCH_SIZE = config['batch_size']
    NUM_WORKERS = config['num_workers']

    PANEL_CONFIG = YAML().load(open(config['panel_config']))
    TOKENIZER = YAML().load(open(config['tokenizer_config']))
    # print(f"Training on datasets: {PANEL_CONFIG['datasets']}")
    # MARKERS_SET = {k for dataset in PANEL_CONFIG['datasets'] for k in PANEL_CONFIG['markers'][dataset]}
    # print(f"Markers set: {MARKERS_SET}")
    # print(f"Number of markers: {len(MARKERS_SET)}")
    # TOKENIZER = {k: v for k, v in zip(MARKERS_SET, range(len(MARKERS_SET)))}
    INV_TOKENIZER = {v: k for k, v in TOKENIZER.items()}

    train_transform = Compose([
        # RandomRotation(180),
        RandomCrop(SIZE)
        # TestCrop(SIZE[0]),
    ])

    test_transform = TestCrop(SIZE[0])

    train_dataset = DatasetFromTIFF(
        panels_config=PANEL_CONFIG,
        split='train',
        marker_tokenizer=TOKENIZER,
        transform=train_transform,
        use_preprocessing=False,
        file_extension="npy"
    )

    test_dataset = DatasetFromTIFF(
        panels_config=PANEL_CONFIG,
        split='test',
        marker_tokenizer=TOKENIZER,
        transform=test_transform,
        use_preprocessing=False,
        file_extension="npy"
    )

    train_batch_sampler = PanelBatchSampler(train_dataset, BATCH_SIZE)
    test_batch_sampler = PanelBatchSampler(test_dataset, BATCH_SIZE, shuffle=False)

    train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=NUM_WORKERS)
    test_dataloader = DataLoader(test_dataset, batch_sampler=test_batch_sampler, num_workers=NUM_WORKERS)
    
    print(f'Training on {len(train_dataloader.dataset)} training samples and {len(test_dataloader.dataset)} test samples')
    print(f'Batch size: {BATCH_SIZE}, Number of workers: {NUM_WORKERS}')

    model_config = {
        'num_channels': len(TOKENIZER),
        'encoder_config': config['encoder'],
        'decoder_config': config['decoder'],
    }

    if config["model_type"] == "EquivariantConvnext":
        from multiplex_model.equivariant_modules import EquivariantMultiplexAutoencoder
        model = EquivariantMultiplexAutoencoder(**model_config).to(device)
    elif config["model_type"] == "Convnext":
        model = MultiplexAutoencoder(**model_config).to(device)

    print(f'Model created with config: {model_config}')
    print(f'Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters')
    print(f'Model: {model}')


    lr = config['lr']
    final_lr = config['final_lr']
    weight_decay = config['weight_decay']
    gradient_accumulation_steps = config['gradient_accumulation_steps']
    epochs = config['epochs']
    total_steps = len(train_dataloader) * epochs // gradient_accumulation_steps
    if 'frac_warmup_steps' in config:
        num_warmup_steps = int(total_steps * float(config['frac_warmup_steps']))
    else:
        num_warmup_steps = int(config['num_warmup_steps'])
    num_annealing_steps = total_steps - num_warmup_steps
    # num_annealing_steps = config['num_annealing_steps']

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_scheduler_with_warmup(optimizer, num_warmup_steps, num_annealing_steps, final_lr=final_lr, type='cosine', base_lr=lr)

    if 'from_checkpoint' in config and config['from_checkpoint']:
        print(f'Loading model from checkpoint: {config["from_checkpoint"]}')
        checkpoint = torch.load(config['from_checkpoint'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Flexible checkpoint loading:
        # - resume_optimizer: load optimizer state (keeps momentum, etc.)
        # - resume_scheduler: load scheduler state (continues LR schedule)
        # Default: fresh optimizer and scheduler with new hyperparameters
        
        if config.get('resume_optimizer', False):
            print('Resuming optimizer state from checkpoint')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print('Using fresh optimizer with new hyperparameters')
        
        if config.get('resume_scheduler', False):
            print('Resuming scheduler state from checkpoint')
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # Important: Update the scheduler's optimizer reference if we created a fresh optimizer
            if not config.get('resume_optimizer', False):
                scheduler.optimizer = optimizer
                print('Updated scheduler to use the new optimizer')
        else:
            print('Using fresh scheduler with new learning rate schedule')
        
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resuming training from epoch {start_epoch}')
    else:
        start_epoch = 0

    
    run["slurm/job_id"] = SLURM_JOB_ID
    # run["sys/run_name"] = run_name

    min_channels_frac = config.get('min_channels_frac', 0.5)
    min_channels_frac_dict = config.get('min_channels_frac_dict')

    parameters = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "lr": lr,
        "weight_decay": weight_decay,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "epochs": epochs,
        "num_warmup_steps": num_warmup_steps,
        "num_annealing_steps": num_annealing_steps,
        "model_config": stringify_unsupported(model_config),
        "min_channels_frac": min_channels_frac,
        # "min_channels_frac_dict": stringify_unsupported(min_channels_frac_dict) if min_channels_frac_dict else None,
    }
    if config.get("from_checkpoint"):
        parameters["from_checkpoint"] = config["from_checkpoint"]
    run["parameters"] = parameters

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
        save_checkpoint_every=config['save_checkpoint_freq'],
        marker_names_map=INV_TOKENIZER,
    )

    run.stop()
