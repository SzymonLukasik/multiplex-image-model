"""Pre-launch checks for the transition-nonlinearity ablation of V3-256
(equiv-v3-int64-zs-hn-f256-noact; configs/README_f256_noact.md).

    CPU (bury):  ONLY=a,b,c  python tests_noact/test_noact_v3_256.py
    GPU (szary): sbatch scripts/szary/test_noact_v3_256.sh        (runs a,b,c,d)

a  Default flags rebuild V3-256 exactly: same state-dict keys and shapes as the
   reference config, and the same initial values under the same seed.
b  The noact model differs from V3-256 ONLY by the removed activation parameters:
   the state dict equals the reference minus the `acts.*.log_bias` entries of the
   MA and PM encoders and `encoder.act.log_bias`; the decoder is identical.
   Counts and names of the removed tensors are printed.
c  Equivariance of the randomly initialised noact model: encoder latent under the
   8 D4 transforms of a 64x64 input (fiber action resolved as in the Fourier test),
   reconstruction under 90 deg, and the 45 deg error. fp32; TF32 off on GPU.
d  (GPU) peak memory and s/100 steps at batch 4, 128 px input (64 internal),
   47 markers, bf16 train step, noact vs V3-256.
"""
import copy
import os
import sys
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF
from ruamel.yaml import YAML

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import escnn.nn as e2nn  # noqa: E402

from multiplex_model.losses import beta_nll_loss  # noqa: E402
from multiplex_model.utils import TrainingConfig, ClampWithGrad  # noqa: E402
from train_masked_model import build_model  # noqa: E402

CFG_REF = "configs/train_equivariant_v3_zs_hn_f256_szary_config.yaml"
CFG_NEW = "configs/train_equivariant_v3_zs_hn_f256_noact_szary_config.yaml"
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cuda":
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
ONLY = set(os.environ.get("ONLY", "a,b,c,d").split(","))
yaml = YAML(typ="safe")
cfg_ref = TrainingConfig(**yaml.load(open(CFG_REF)))
cfg_new = TrainingConfig(**yaml.load(open(CFG_NEW)))
NC = len(cfg_ref.tokenizer_config)
assert cfg_ref.model_type == cfg_new.model_type


def model_cfg(cfg):
    enc = copy.deepcopy(cfg.encoder_config.model_dump())
    return {"num_channels": NC, "encoder_config": enc,
            "decoder_config": copy.deepcopy(cfg.decoder_config.model_dump())}


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def rel(a, b, m=None):
    if m is not None:
        a, b = a * m, b * m
    return float((a - b).norm() / b.norm())


# the two configs must differ in exactly the two flags (plus bookkeeping keys)
d_ref, d_new = cfg_ref.encoder_config.model_dump(), cfg_new.encoder_config.model_dump()
diff_keys = {k for k in set(d_ref) | set(d_new) if d_ref.get(k) != d_new.get(k)}
print(f"encoder config keys that differ: {sorted(diff_keys)}")
assert diff_keys == {"pool_use_act", "use_hyperkernel_act"}, diff_keys
assert cfg_ref.decoder_config.model_dump() == cfg_new.decoder_config.model_dump()
assert d_new["pool_use_act"] is False and d_new["use_hyperkernel_act"] is False

# ---------------------------------------------------------------- a
if "a" in ONLY:
    print("\n== a: default flags rebuild V3-256 ==")
    torch.manual_seed(0)
    m_ref = build_model(model_cfg(cfg_ref), cfg_ref.model_type)
    enc_plain = model_cfg(cfg_ref)
    enc_plain["encoder_config"].pop("pool_use_act", None)   # module default must equal the config value
    torch.manual_seed(0)
    m_def = build_model(enc_plain, cfg_ref.model_type)
    s1, s2 = m_ref.state_dict(), m_def.state_dict()
    same = list(s1) == list(s2) and all(s1[k].shape == s2[k].shape and torch.equal(s1[k], s2[k]) for k in s1)
    print(f"  config-flag model == module-default model (keys, shapes, values under seed 0): {same}")
    print(f"  V3-256 trainable params: {n_params(m_ref)/1e6:.3f} M (encoder {n_params(m_ref.encoder)/1e6:.3f} M, "
          f"decoder {n_params(m_ref.decoder)/1e6:.3f} M)")
    del m_def

# ---------------------------------------------------------------- b
if "b" in ONLY or "c" in ONLY or "d" in ONLY:
    if "a" not in ONLY:
        torch.manual_seed(0)
        m_ref = build_model(model_cfg(cfg_ref), cfg_ref.model_type)
    torch.manual_seed(0)
    m_new = build_model(model_cfg(cfg_new), cfg_new.model_type)
if "b" in ONLY:
    print("\n== b: noact == V3-256 minus the transition activations ==")
    s_ref, s_new = m_ref.state_dict(), m_new.state_dict()
    only_ref = [k for k in s_ref if k not in s_new]
    only_new = [k for k in s_new if k not in s_ref]
    shape_ok = all(s_ref[k].shape == s_new[k].shape for k in s_new)
    removed_params = sum(s_ref[k].numel() for k in only_ref)
    print(f"  keys only in V3-256 ({len(only_ref)}, {removed_params} params):")
    for k in only_ref:
        print(f"    {k}  {tuple(s_ref[k].shape)}")
    print(f"  keys only in noact: {only_new}")
    print(f"  shared keys have identical shapes: {shape_ok}")
    print(f"  decoder state dicts identical in keys and shapes: "
          f"{[(k, tuple(v.shape)) for k, v in m_ref.decoder.state_dict().items()] == [(k, tuple(v.shape)) for k, v in m_new.decoder.state_dict().items()]}")
    # every removed key must belong to one of the five NormNonLinearity modules (log_bias parameter, eps / indices buffers)
    act_like = [k for k in only_ref if not ((".acts." in k) or (".encoder.act." in k) or k.startswith("encoder.act."))]
    print(f"  removed keys that do NOT belong to an activation module: {act_like}")
    print(f"  removed trainable parameters: {sum(s_ref[k].numel() for k in only_ref if k.endswith('log_bias'))}")
    print(f"  noact trainable params: {n_params(m_new)/1e6:.3f} M "
          f"(diff vs V3-256 {n_params(m_new) - n_params(m_ref):+d})")
    enc = m_new.encoder
    print(f"  encoder.act is {type(enc.act).__name__}; MA pool_use_act={enc.marker_agnostic_encoder.pool_use_act}; "
          f"PM pool_use_act={enc.pan_marker_encoder.pool_use_act}; "
          f"MA has 'acts': {hasattr(enc.marker_agnostic_encoder, 'acts')}; PM has 'acts': {hasattr(enc.pan_marker_encoder, 'acts')}")
    print(f"  decoder bottleneck_act: {type(m_new.decoder.bottleneck_act).__name__} "
          f"(function {m_new.decoder.norm_act_function}); refine acts: {len(m_new.decoder.refine_acts)} (unchanged)")
    assert not only_new and shape_ok and not act_like

# ---------------------------------------------------------------- c
if "c" in ONLY:
    print(f"\n== c: equivariance of the random-init noact model ({dev}, fp32, eval) ==")
    model = m_new.to(dev).eval()
    C = 6 if dev == "cpu" else 12
    torch.manual_seed(1)
    H = 64
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, H), indexing="ij")
    disk = ((yy ** 2 + xx ** 2) <= 1.0).float().to(dev)
    img = torch.rand(1, C, H, H, device=dev) * disk
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
        print(f"  max latent rel err over D4 = {max(lat_err.values()):.2e} (pass < 1e-5: {max(lat_err.values()) < 1e-5}); "
              f"max recon rel err = {max(rec_err.values()):.2e}")
        m7 = ((yy ** 2 + xx ** 2) <= 0.49).float().to(dev)
        for ang in (45.0, 15.0):
            xr = TF.rotate(img, ang, interpolation=TF.InterpolationMode.BILINEAR)
            rt = torch.sigmoid(model(xr, ids, ids)["output"][..., 0])
            rref = TF.rotate(rr, ang, interpolation=TF.InterpolationMode.BILINEAR)
            fl = TF.rotate(TF.rotate(rr, ang, interpolation=TF.InterpolationMode.BILINEAR), -ang,
                           interpolation=TF.InterpolationMode.BILINEAR)
            print(f"  {ang:.0f} deg: recon rel-L2 (central disk) = {rel(rt, rref, m7):.3f} (interp floor {rel(fl, rr, m7):.3f})")
        mn = m_ref.to(dev).eval()
        rrn = torch.sigmoid(mn(img, ids, ids)["output"][..., 0])
        xr = TF.rotate(img, 45.0, interpolation=TF.InterpolationMode.BILINEAR)
        rtn = torch.sigmoid(mn(xr, ids, ids)["output"][..., 0])
        print(f"  (V3-256, random init) 45 deg recon rel-L2 = "
              f"{rel(rtn, TF.rotate(rrn, 45.0, interpolation=TF.InterpolationMode.BILINEAR), m7):.3f}; "
              f"90 deg = {rel(torch.sigmoid(mn(d4(img,1,False), ids, ids)['output'][..., 0]), d4(rrn,1,False)):.2e}")
        # how much of the encoder's input to the first PM block is now untouched by any nonlinearity:
        # fraction of exactly-zero field vectors at the stem output (V3-256 zeroes ~half of them)
        def zero_frac(m):
            enc = m.encoder
            x = img.reshape(-1, 1, H, H)
            g = e2nn.GeometricTensor(x, enc.marker_agnostic_encoder.input_type)
            ma = enc.marker_agnostic_encoder
            g = ma.poolings[0](g)
            if ma.use_norm:
                g = ma.pool_norms[0](g)
            if ma.pool_use_act:
                g = ma.acts[0](g)
            t = g.tensor
            K = len(ma.poolings[0].out_type) if hasattr(ma.poolings[0], "out_type") else t.shape[1] // 10
            v = t.reshape(t.shape[0], -1, 10, *t.shape[-2:]).norm(dim=2)
            return float((v == 0).float().mean())
        print(f"  fraction of exactly-zero stem field vectors (random init): V3-256 {zero_frac(mn):.3f}, noact {zero_frac(model):.3f}")
    model.cpu(); m_ref.cpu()

# ---------------------------------------------------------------- d
if "d" in ONLY and dev == "cuda":
    print("\n== d: memory / speed: bs 4, 128 px input (64 internal), 47 markers, bf16 train step ==")
    for name, cfg in [("V3-256", cfg_ref), ("noact", cfg_new)]:
        torch.manual_seed(0)
        m = build_model(model_cfg(cfg), cfg.model_type).to(dev).train()
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
            print(f"  {name:<8} {dt:6.1f} s/100 steps  peak {pk:5.1f} GiB  -> ~{dt/100*9128*2/3600:.2f} h/epoch "
                  f"at 9128 optimizer steps x accumulation 2 (1 GPU)  (loss {loss.item():.3f})", flush=True)
        except torch.OutOfMemoryError:
            print(f"  {name:<8} OOM at batch 4", flush=True)
        del m, opt
        torch.cuda.empty_cache()
print("\nDONE")
