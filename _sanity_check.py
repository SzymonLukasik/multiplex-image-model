"""Offline sanity check (no real data): build the equivariant hook, strict-load
the registered checkpoint, run encode_images on synthetic crops."""
import sys, time, yaml, torch, numpy as np

PROJ = "/net/tscratch/people/plgslukasik/immu-vis/multiplex-image-model/celltyping-downstream/celltyping"
sys.path.insert(0, PROJ)

from core.models.immuvis_equivariant import setup_immuvis_equivariant

reg = yaml.safe_load(open(f"{PROJ}/core/models/registry.yaml"))["models"]["equiv_convnext_v2"]
reg.pop("setup_func")
reg["scheme"] = "patch"
reg["batch_size"] = 4
reg["dataset_name"] = "nsclc2-panel1"

t = time.time()
model, prepare_fn, get_channels, compute_fn, device = setup_immuvis_equivariant(**reg)
print(f"[ok] model built + checkpoint strict-loaded in {time.time()-t:.1f}s on {device}")

ch = get_channels(None, None)
n = len(ch)
for size in (32, 64):
    raw = torch.rand(n, size, size) * 8.0          # synthetic raw crop (C,H,W)
    prepped, _ = prepare_fn(raw, None, "t0", None)
    print(f"  prepare_fn: {tuple(raw.shape)} -> {tuple(prepped.shape)} "
          f"[{prepped.min():.3f},{prepped.max():.3f}]")
    crops = [prepped.clone() for _ in range(4)]
    chans = [ch for _ in range(4)]
    try:
        _, tok, _, _ = compute_fn(model, crops, chans, None, device, batch_size=4)
        print(f"  [ok] encode_images @ {size}x{size}: embeddings {tuple(tok.shape)}")
    except Exception as e:
        print(f"  [FAIL] @ {size}x{size}: {type(e).__name__}: {e}")
print("SANITY DONE")
