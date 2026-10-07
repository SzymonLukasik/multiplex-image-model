"""Time one optimiser micro-step (fwd + bwd + AdamW) on synthetic input, and project
hours per epoch. Complements _profile_gpu_memory.py, which measures memory only.

Usage:  python _time_train_step.py <config.yaml> [--batch 2] [--channels 40] [--steps 30]
        [--iters-per-epoch 18256]

--iters-per-epoch defaults to 18256 = 36512 zs-hn training samples / batch 2 on one
GPU (the 2-GPU ma4 run logged 9128 iterations/epoch at batch 2 per rank).
Synthetic data excludes data-loading time, so treat the projection as a lower bound.
"""
import argparse
import time

import torch
from ruamel.yaml import YAML

from multiplex_model.losses import beta_nll_loss
from multiplex_model.utils import ClampWithGrad, TrainingConfig
from train_masked_model import build_model

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--batch", type=int, default=None)
ap.add_argument("--channels", type=int, default=40)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--iters-per-epoch", type=int, default=18256)
a = ap.parse_args()

cfg = TrainingConfig(**YAML(typ="safe").load(open(a.config)))
B = a.batch if a.batch is not None else cfg.batch_size
H = cfg.input_image_size[0]
NC = len(cfg.tokenizer_config)
dev = "cuda"

model = build_model(
    {"num_channels": NC, "encoder_config": cfg.encoder_config.model_dump(),
     "decoder_config": cfg.decoder_config.model_dump()}, cfg.model_type
).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
print(f"GPU {torch.cuda.get_device_name(0)} | params {sum(p.numel() for p in model.parameters())/1e6:.1f} M "
      f"| batch {B} | channels {a.channels} | input {H}")


def step():
    x = torch.rand(B, a.channels, H, H, device=dev)
    ids = torch.randint(0, NC, (B, a.channels), device=dev)
    mi, lv = model(x, ids, ids)["output"].unbind(dim=-1)
    loss = beta_nll_loss(x, torch.sigmoid(mi), ClampWithGrad.apply(lv, -15.0, 15.0), beta=cfg.beta)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


for _ in range(a.warmup):
    step()
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
for _ in range(a.steps):
    step()
torch.cuda.synchronize()
s = (time.time() - t0) / a.steps
print(f"{s:.3f} s / micro-step | peak {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB | "
      f"projected {a.iters_per_epoch * s / 3600:.2f} h / epoch (lower bound, no data loading)")
