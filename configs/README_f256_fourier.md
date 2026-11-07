# equiv-v3-f256-fourier-zs-hn: Fourier-ELU ablation of V3-256

Written 2026-10-27, before training started. The pre-registered reading below must not be edited after results arrive.

## Question

Does the norm-based nonlinearity limit reconstruction? Every `NormNonLinearity` in the ENCODER of V3-256 (EquiVis-256) is replaced by escnn's Fourier pointwise ELU, `FourierELU`. It samples each bl_regular(2) field on 8 rotations × 2 flips (16 points), applies ELU and transforms back. Nothing else changes: the decoder, hyperkernel, trivial gate, PixelLN, layer scale, downsampling, readout, recipe and data are all identical.

## Config

`multiplex-image-model-fresh/configs/train_equivariant_v3_ddp_zs_hn_f256_fourier_config.yaml`, diffed against `train_equivariant_v3_ddp_zs_hn_config.yaml`:

```
encoder.pm_embedding_dims             [128, 256, 512] -> [64, 128, 256]   (V3-256 half width, szary diff)
encoder.hyperkernel_config.embedding_dim         128 -> 64                (V3-256 half width, szary diff)
encoder.nonlinearity                                 + fourier_elu        (default 'norm' = current model)
encoder.fourier_N                                    + 8
checkpoints_dir / run_name                           -> checkpoints_equiv_zs_hn_f256_fourier / equiv-v3-f256-fourier-zs-hn
```

The rest is unchanged:
- ma_layers_blocks [1], maximum_frequency 2, internal 64.
- Decoder: freq [2,2,1], fields [128,96,64], transposed conv, n_softplus, post_refine_norm.
- β-NLL with β 0.5; LR 5e-4 → 1e-5 cosine with 10% warm-up; weight decay 1e-4; 200 epochs.
- 2 GPUs × 4 = effective batch 8; rotflip augmentation; zero-shot hn panels.
- Sampler seed 42, the default. This repo sets no torch seed, so initialisation is unseeded, as in every reference run.
- Checkpoints every 10 epochs.

No throughput settings differ from V3-256. cuDNN autotuning was benchmarked at +4% on V3-ma4 but −6% on V3-ma1, which is noise, so it's left off. `torch.compile` breaks the escnn graph 52 times and runs 4× slower.

Code: `multiplex_model/modules/equivariant/equivariant_modules_v2.py` (`FourierNormAct`, `make_act`, and the `nonlinearity` / `fourier_N` kwargs plumbed through `EquivariantMultiplexImageEncoder` → `EquivariantConvNeXtEncoder` (MA and PM) → `EscnnConvNeXtBlocks` → `BLConvNeXtBlock`). Both keys are stored in the checkpoint's `model_config` through `get_architecture_config`. The encoder config is `extra="allow"`, so no schema change was needed.

## Pre-launch tests (`tests_fourier/test_fourier_v3_256.py`; logs/test_fourier256_3402599 and _3402841)

- **3a:** `FourierNormAct` equals `FourierELU` on the re-wrapped tensor, max |diff| 0.0. The input representation is bl_regular(2), with irreps [(0,0),(1,0),(1,1),(1,1),(1,2),(1,2)] and 16 grid points per field.
- **3b** (random initialisation, fp32, TF32 off):
  - The D4 latent relative error is at most 1.0e-6 and the reconstruction relative error at most 3.0e-7, across all 8 elements; the norm model gives 2.7e-7 at 90°.
  - With TF32 on (the PyTorch default for cuDNN convolutions), both models sit at about 5–8e-5. That is a precision floor.
  - The 45° reconstruction relative L2 in the central disk is 0.182 for Fourier vs 0.123 for norm, with an interpolation floor of 0.048. These are random-init values and not comparable with the trained V3's 7–10%.
- **3c** (batch 4, 128 px input at 64 internal, 47 markers, bf16 train step, 1 A100):

  | | s/100 steps | peak memory |
  |---|---|---|
  | norm | 39.4 | 19.1 GiB |
  | fourier N=8 | 40.0 | 19.5 GiB |
  | fourier N=6 | 39.2 | 19.3 GiB |

  N=8 fits at 2 × 4, so no fallback was needed.
- **3d:** `nonlinearity: norm` rebuilds the pre-change encoder exactly: identical keys and shapes, and identical initial values under the same seed. The full model has 122.99M trainable parameters (encoder 33.51M, decoder 89.48M).
  - Fourier has 122.98M, 7,760 fewer: the norm activations carry learnable biases, while the Fourier activations hold only fixed `A` / `Ainv` buffers.
  - The decoder state-dict keys are identical.

## Evaluation

Zero-shot leave-one-out on the 149 hn test patches with `run_validation.py`, exactly as for the reference runs:
- 128 px, patch_mask_ratio 0, scored at 128, with the per-image `*_full.csv` holding the sufficient statistics.
- Epochs: e29, e49, e79, then every 10 epochs.
- The evaluation runs automatically after each training link (`scripts/eval_f256_fourier_pending.sh`). The table comes from `compare_f256_fourier.py`.

References:
- V3-256 (szary), per-image Pearson / MSE: e29 .6059 / .00746, e49 .6128 / .00693, e79 .6242 / .00648; best so far .6303 at e149.
- V3-256 validation: NLL −5.97 at e10, −6.37 at e24, −6.60 at e46, −6.70 at e69; MSE .00399 at e24 and .00384 at e28.
- V3-512 (Athena, same definition), Pearson / MSE: e29 .5990 / .00807, e49 .6236 / .00618, e79 .6311 / .00586, e199 .6388 / .00544.

## Pre-registered reading

- **Primary:** per-image LOO Pearson at e79, Fourier minus V3-256.
  - **Gain** if ≥ +.004, about half the V3-256 vs V3-512 gap.
  - **Null** if within ±.004.
  - **Loss** if below −.004.
  - The paired per-image z is reported as well, but the decision uses the threshold: single-run differences below .004 are within training-seed noise.
- **Secondary:** validation MSE at matched epochs, and the LOO MSE.
- **Downstream** (cell typing, clinical) runs on szary afterwards on the e79 and final checkpoints, only if the primary is a gain or the LOO MSE improves by 3% or more.

No other modification is combined with this change in this run.

## Addendum 2026-10-27 13:10 (before any LOO result existed)

An extra exploratory LOO evaluation was added at e9 (EPOCH 10), compared against V3-512 e9 (.5591 / .00885) and against V3-256 e9 if a szary number is supplied. It does not change the pre-registered reading above: the decision is still made at e79.
V3-256 e9 from szary: per-image .5546, pooled .6947, MSE .00848.

From 2026-10-27 18:45 the run self-chains (`scripts/selfchain_lib.sh`, at most one queued job per run) and the LOO evaluations run inside each training link, before training starts, instead of as separate jobs. The protocol is unchanged.
