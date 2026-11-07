"""Pre-launch checks for the Fourier-ELU V3-256 run (equiv-v3-f256-fourier-zs-hn).

    sbatch ../scripts/run_test_fourier_v3_256.sh

3a  FourierNormAct == FourierELU applied to the re-wrapped tensor.
3b  Randomly initialised full model (fourier_elu): encoder latent under the 8 D4
    transforms of a 64x64 input (fiber action resolved by test, as in
    probe_equivariance_fresh.py), reconstruction under 90 deg, and the 45 deg error.
3c  Peak memory and s/100 steps, batch 4, 128 px input (64 internal), 47 markers,
    bf16 train step, for 'norm' vs 'fourier_elu' (and fourier_N=6 if requested).
3d  nonlinearity='norm' rebuilds the pre-change V3-256 encoder bit for bit
    (same keys, shapes and, under the same seed, the same initial values).
"""
import copy
import importlib.util
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from ruamel.yaml import YAML

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import escnn.nn as e2nn  # noqa: E402

from multiplex_model.losses import beta_nll_loss  # noqa: E402
from multiplex_model.utils import TrainingConfig, ClampWithGrad  # noqa: E402
from multiplex_model.modules.equivariant.equivariant_modules_v2 import FourierNormAct  # noqa: E402
from train_masked_model import build_model  # noqa: E402

CFG = "configs/train_equivariant_v3_ddp_zs_hn_f256_fourier_config.yaml"
dev = "cuda"
if os.environ.get("NO_TF32") == "1":
    # PyTorch lets cuDNN use TF32 for convs by default (~1e-4 relative error),
    # which floors the exactness test above 1e-5 for ANY model. Full fp32 here.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    print("TF32 disabled (cudnn + matmul)")
ONLY = set(os.environ.get("ONLY", "3a,3b,3c,3d").split(","))
cfg = TrainingConfig(**YAML(typ="safe").load(open(CFG)))
NC = len(cfg.tokenizer_config)


def model_cfg(kind, N=8):
    enc = copy.deepcopy(cfg.encoder_config.model_dump())
    enc["nonlinearity"], enc["fourier_N"] = kind, N
    return {"num_channels": NC, "encoder_config": enc,
            "decoder_config": copy.deepcopy(cfg.decoder_config.model_dump())}


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def rel(a, b, m=None):
    if m is not None:
        a, b = a * m, b * m
    return float((a - b).norm() / b.norm())


# ---------------------------------------------------------------- 3a
print("== 3a unit ==")
import escnn  # noqa: E402
gs = escnn.gspaces.flipRot2dOnR2(N=-1, maximum_frequency=4)
bl = gs.fibergroup.bl_regular_representation(2)
ft = e2nn.FieldType(gs, [bl] * 16)
act = FourierNormAct(ft, 8).to(dev)
x = torch.randn(4, ft.size, 17, 17, device=dev)
x0 = x.clone()
y = act(e2nn.GeometricTensor(x, ft))
ref = act.act(e2nn.GeometricTensor(x0.clone(), act.act.in_type))
print(f"  irreps {bl.irreps}; grid points per field = {act.act.A.shape[0] if hasattr(act.act, 'A') else '?'}")
print(f"  max|FourierNormAct - FourierELU| = {(y.tensor - ref.tensor).abs().max().item():.3e}; "
      f"input untouched: {torch.equal(x, x0)}; out type is in type: {y.type == ft}")

# ---------------------------------------------------------------- 3d
print("\n== 3d norm rebuild vs pre-change module ==")
spec = importlib.util.spec_from_file_location(
    "ref_v2", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ref_equivariant_modules_v2_norm_only.py"))
ref_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref_mod)
enc_cfg = copy.deepcopy(cfg.encoder_config.model_dump())
enc_cfg.pop("internal_image_size", None); enc_cfg.pop("nonlinearity"); enc_cfg.pop("fourier_N")
enc_cfg["output_trivial"] = False; enc_cfg.pop("output_scalars", None)
torch.manual_seed(0)
ref_enc = ref_mod.EquivariantMultiplexImageEncoder(num_channels=NC, **enc_cfg)
torch.manual_seed(0)
new_norm = build_model(model_cfg("norm"), cfg.model_type)     # full model; encoder built first
sd_r, sd_n = ref_enc.state_dict(), new_norm.encoder.state_dict()
same_keys = list(sd_r) == list(sd_n)
same_shape = same_keys and all(sd_r[k].shape == sd_n[k].shape for k in sd_r)
same_vals = same_shape and all(torch.equal(sd_r[k], sd_n[k]) for k in sd_r)
print(f"  encoder keys identical: {same_keys}; shapes identical: {same_shape}; "
      f"initial values identical under seed 0: {same_vals}")
print(f"  trainable params, full model, norm: {n_params(new_norm)/1e6:.2f} M "
      f"(encoder {n_params(new_norm.encoder)/1e6:.2f} M, decoder {n_params(new_norm.decoder)/1e6:.2f} M)")
torch.manual_seed(0)
four = build_model(model_cfg("fourier_elu"), cfg.model_type)
kn, kf = set(new_norm.state_dict()), set(four.state_dict())
only_n = sorted(kn - kf); only_f = sorted(kf - kn)
print(f"  trainable params, full model, fourier_elu: {n_params(four)/1e6:.2f} M "
      f"(diff {n_params(four) - n_params(new_norm):+d})")
print(f"  state-dict keys only in norm ({len(only_n)}): e.g. {only_n[:3]}")
print(f"  state-dict keys only in fourier ({len(only_f)}): e.g. {only_f[:3]}")
print(f"  decoder state dicts identical in keys: "
      f"{list(new_norm.decoder.state_dict()) == list(four.decoder.state_dict())}")
del ref_enc, new_norm

# ---------------------------------------------------------------- 3b
print("\n== 3b equivariance, random-init fourier_elu model (fp32, eval) ==")
model = four.to(dev).eval()
C = 12
torch.manual_seed(1)
H = 64
yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, H), indexing="ij")
disk = ((yy ** 2 + xx ** 2) <= 1.0).float().to(dev)
img = torch.rand(1, C, H, H, device=dev) * disk          # circle-masked input
ids = torch.randperm(NC)[:C].unsqueeze(0).to(dev)


def d4(x, k, flip):
    y = torch.rot90(x, k=k % 4, dims=(-2, -1))
    return torch.flip(y, dims=[-1]) if flip else y


with torch.no_grad():
    zr = model.encode(img, ids)["output"]
    ftype = model._latent_type
    rr = torch.sigmoid(model(img, ids, ids)["output"][..., 0])
    G = ftype.gspace.fibergroup
    in_t = e2nn.FieldType(ftype.gspace, C * [ftype.gspace.trivial_repr])
    lat_err, rec_err = {}, {}
    for k in range(4):
        for flip in (False, True):
            if k == 0 and not flip:
                continue
            want = d4(img, k, flip)
            best, be = None, 1e9
            for fb in (0, 1):
                for s in (1, -1):
                    g = G.element((fb, s * k * np.pi / 2))
                    e = (in_t.transform(img, g) - want).abs().max().item()
                    if e < be:
                        best, be = g, e
            zt = model.encode(want, ids)["output"]
            zi = torch.flip(zt, dims=[-1]) if flip else zt
            zi = torch.rot90(zi, k=-k % 4, dims=(-2, -1))
            zi = ftype.transform_fibers(zi, ~best)
            lat_err[(k, flip)] = rel(zi, zr)
            rt = torch.sigmoid(model(want, ids, ids)["output"][..., 0])
            rec_err[(k, flip)] = rel(rt, d4(rr, k, flip))
    for kf_, e in lat_err.items():
        print(f"  k={kf_[0]*90:3d} flip={int(kf_[1])}  latent rel err {e:.2e}   recon rel err {rec_err[kf_]:.2e}")
    print(f"  max latent rel err over D4 = {max(lat_err.values()):.2e}  "
          f"(pass < 1e-5: {max(lat_err.values()) < 1e-5}); "
          f"max recon rel err = {max(rec_err.values()):.2e}")
    # 45 degrees: bilinear rotation of a circle-masked input; compare recon to
    # the rotated reference inside the central disk (r = 0.7).
    m7 = ((yy ** 2 + xx ** 2) <= 0.49).float().to(dev)
    for ang in (45.0, 15.0):
        xr = TF.rotate(img, ang, interpolation=TF.InterpolationMode.BILINEAR)
        rt = torch.sigmoid(model(xr, ids, ids)["output"][..., 0])
        rref = TF.rotate(rr, ang, interpolation=TF.InterpolationMode.BILINEAR)
        fl = TF.rotate(TF.rotate(rr, ang, interpolation=TF.InterpolationMode.BILINEAR), -ang,
                       interpolation=TF.InterpolationMode.BILINEAR)
        print(f"  {ang:.0f} deg: recon rel-L2 (central disk) = {rel(rt, rref, m7):.3f} "
              f"(interp floor {rel(fl, rr, m7):.3f})")
    # same check for the norm model, for reference
    torch.manual_seed(0)
    mn = build_model(model_cfg("norm"), cfg.model_type).to(dev).eval()
    rrn = torch.sigmoid(mn(img, ids, ids)["output"][..., 0])
    xr = TF.rotate(img, 45.0, interpolation=TF.InterpolationMode.BILINEAR)
    rtn = torch.sigmoid(mn(xr, ids, ids)["output"][..., 0])
    print(f"  (norm model, random init) 45 deg recon rel-L2 = "
          f"{rel(rtn, TF.rotate(rrn, 45.0, interpolation=TF.InterpolationMode.BILINEAR), m7):.3f}; "
          f"90 deg = {rel(torch.sigmoid(mn(d4(img,1,False), ids, ids)['output'][..., 0]), d4(rrn,1,False)):.2e}")
    del mn
del model, four
torch.cuda.empty_cache()

# ---------------------------------------------------------------- 3c
if "3c" not in ONLY:
    sys.exit(0)
print("\n== 3c memory / speed: bs 4, 128 px input (64 internal), 47 markers, bf16 train step ==")
Ns = [int(a) for a in os.environ.get("FOURIER_NS", "8").split(",")]
for kind, N in [("norm", 8)] + [("fourier_elu", n) for n in Ns]:
    torch.manual_seed(0)
    m = build_model(model_cfg(kind, N), cfg.model_type).to(dev).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-5)
    B, C, Hh = 4, 47, 128
    x = torch.rand(B, C, Hh, Hh, device=dev)
    ids = torch.stack([torch.randperm(NC)[:C] for _ in range(B)]).to(dev)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(x, ids, ids)["output"]
        mi, lv = out.float().unbind(dim=-1)
        loss = beta_nll_loss(x, torch.sigmoid(mi), ClampWithGrad.apply(lv, -15., 15.), beta=cfg.beta)
        loss.backward()
        opt.step()
        return loss

    torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(5):
            step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            loss = step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        pk = torch.cuda.max_memory_allocated() / 2**30
        print(f"  {kind:<12} N={N}  {dt:6.1f} s/100 steps  peak {pk:5.1f} GiB  "
              f"-> ~{dt/100*9126/3600:.2f} h/epoch at 9126 steps/rank  (loss {loss.item():.3f})", flush=True)
    except torch.OutOfMemoryError:
        print(f"  {kind:<12} N={N}  OOM at batch 4", flush=True)
    del m, opt
    torch.cuda.empty_cache()
