"""Attribute GPU memory in the equivariant model — where does it actually go?

Answers three questions a plain `nvidia-smi` cannot:
  1. params vs optimizer state vs activations,
  2. which STAGE holds the activations (MA encoder / hyperkernel / PM stages / decoder),
  3. how it scales with the panel's channel count C — the real OOM driver, since the
     marker-agnostic encoder reshapes to (B*C, 1, H, W) and runs at full internal
     resolution, so a 47-marker panel costs ~1.7x a 28-marker one at the same batch.

Usage:  python _profile_gpu_memory.py <config.yaml> [C ...]
"""
import sys
import torch
from ruamel.yaml import YAML

from multiplex_model.utils import TrainingConfig, ClampWithGrad
from multiplex_model.losses import beta_nll_loss
from train_masked_model import build_model

GB = 1024 ** 3


def prof(cfg_path, channel_counts, batch=None, internal=None):
    cfg = TrainingConfig(**YAML(typ="safe").load(open(cfg_path)))
    enc = cfg.encoder_config.model_dump()
    if internal is not None:
        enc["internal_image_size"] = internal
    NC = len(cfg.tokenizer_config)
    H = cfg.input_image_size[0]
    B = batch if batch is not None else cfg.batch_size
    dev = "cuda"

    print("=" * 78)
    print(f"CONFIG {cfg_path}")
    print(f"  pm_embedding_dims={enc['pm_embedding_dims']}  hk_embed={enc['hyperkernel_config']['embedding_dim']}"
          f"  internal={enc.get('internal_image_size')}  input={H}  batch={B}")
    print("=" * 78)

    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    model = build_model(
        {"num_channels": NC, "encoder_config": enc,
         "decoder_config": cfg.decoder_config.model_dump()}, cfg.model_type
    ).to(dev)
    params_gb = (torch.cuda.memory_allocated() - base) / GB
    nparam = sum(p.numel() for p in model.parameters())
    print(f"\n  parameters        : {nparam/1e6:8.1f} M   -> {params_gb:6.2f} GiB resident")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # ---- per-stage activation attribution via forward hooks ----------------
    marks, handles = [], []
    def mk(name):
        def hook(m, i, o):
            marks.append((name, torch.cuda.memory_allocated()))
        return hook
    e = model.encoder
    targets = [("MA encoder", e.marker_agnostic_encoder)]
    if getattr(e, "ma_to_hk_adapter", None) is not None:
        targets.append(("MA->HK adapter", e.ma_to_hk_adapter))
    targets += [("hyperkernel", e.hyperkernel), ("act", e.act),
                ("PM encoder", e.pan_marker_encoder), ("decoder", model.decoder)]
    for n, m in targets:
        handles.append(m.register_forward_hook(mk(n)))

    for C in channel_counts:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        marks.clear()
        pre = torch.cuda.memory_allocated()
        x = torch.rand(B, C, H, H, device=dev)
        ids = torch.randint(0, NC, (B, C), device=dev)

        try:
            out = model(x, ids, ids)["output"]
            fwd_peak = torch.cuda.max_memory_allocated()
            mi, lv = out.unbind(dim=-1)
            loss = beta_nll_loss(x, torch.sigmoid(mi), ClampWithGrad.apply(lv, -15., 15.), beta=cfg.beta)
            loss.backward()
            bwd_peak = torch.cuda.max_memory_allocated()
            opt.step(); opt.zero_grad(set_to_none=True)
            opt_peak = torch.cuda.max_memory_allocated()

            print(f"\n  --- C = {C} markers (B*C = {B*C} images through the MA stage) ---")
            prev = pre
            for n, v in marks:
                print(f"      after {n:<16} cumulative activations {(v-pre)/GB:6.2f} GiB "
                      f"(+{(v-prev)/GB:5.2f})")
                prev = v
            print(f"      PEAK forward   {fwd_peak/GB:6.2f} GiB   "
                  f"(activations {(fwd_peak-pre)/GB:.2f} above params)")
            print(f"      PEAK fwd+bwd   {bwd_peak/GB:6.2f} GiB")
            print(f"      PEAK +opt step {opt_peak/GB:6.2f} GiB   <- this is what must fit")
        except torch.cuda.OutOfMemoryError:
            print(f"\n  --- C = {C}: OOM (needed more than the card has) ---")
            torch.cuda.empty_cache()
        finally:
            for v in ("out", "loss", "mi", "lv", "x", "ids"):
                if v in dir(): pass
            torch.cuda.empty_cache()

    for h in handles:
        h.remove()
    del model, opt
    torch.cuda.empty_cache()


if __name__ == "__main__":
    total = torch.cuda.get_device_properties(0).total_memory / GB
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  {total:.1f} GiB total\n")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--channels", type=int, nargs="+", default=[28, 40, 47])
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--internal", type=int, default=None)
    a = ap.parse_args()
    prof(a.config, a.channels, a.batch, a.internal)
