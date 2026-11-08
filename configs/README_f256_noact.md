# equiv-v3-int64-zs-hn-f256-noact: transition-nonlinearity ablation of V3-256

Written 2026-11-08, before training started. The pre-registered reading below must not be edited after results arrive.

## Question

The base ImmuVis encoder follows the ConvNeXt convention that every transition is linear plus normalisation: stem conv → LayerNorm, LayerNorm → stride-2 conv, hyperkernel → LayerNorm. Its only nonlinearities are the GELUs inside the blocks. EquiVis V3 puts a `NormNonLinearity` at five additional places in the encoder: after the stem conv, after the hyperkernel, at the stage-1 entry (1×1 R2Conv) and after both stride-2 downsamples. None of these is required for equivariance (a steerable conv followed by `EquivariantPixelLN` is already equivariant), so they are a second architectural asymmetry between EquiVis and ImmuVis besides the stem depth (4 blocks vs 1), this one in EquiVis's favour.

Do these five activations matter? This run removes all five and changes nothing else. The 13 in-block `NormAct` + trivial-gate pairs, `EquivariantPixelLN` at every transition, the hyperkernel, the decoder (including its own `NormAct`s, which the vanilla progressive-decoder control also has as GELUs), the readout, recipe and data are identical to V3-256.

Outcomes and what each would mean:

- **Null:** the extra activations are not what makes EquiVis work; the confound is closed and the ImmuVis-vs-EquiVis comparison does not need a correction for it.
- **Loss:** the extra activations contribute measurably; the thesis must list them next to the stem depth as an asymmetry and quantify it from this run.
- **Collapse** (flat reconstruction, as the early decoders did): the norm nonlinearity alone inside the blocks is too weak to train this network, which is itself a result about the nonlinearity, and the strongest form of "loss".
- **Gain:** unexpected; would mean the transition activations hurt, most likely by thresholding the stem fields before the first block.

## Config

`configs/train_equivariant_v3_zs_hn_f256_noact_szary_config.yaml`, diffed against `train_equivariant_v3_zs_hn_f256_szary_config.yaml` (V3-256):

```
encoder.pool_use_act                 true -> false   (no NormAct after stem conv, stage-1 entry, downsamples)
encoder.use_hyperkernel_act           (new) false    (no NormAct after the encoder hyperkernel; default true = V3-256)
checkpoints_dir / run_name / tags     -> checkpoints_equiv_zs_hn_f256_noact / equiv-v3-int64-zs-hn-f256-noact
```

The rest is unchanged:
- ma_layers_blocks [1], ma 16 fields, pm [64, 128, 256], hyperkernel 64, maximum_frequency 2, internal 64, use_norm true, gating, layer scale 1e-6, gate bias 1.0.
- Decoder: freq [2,2,1], fields [128,96,64], transposed conv, n_softplus, post_refine_norm, in_block_norm.
- β-NLL with β 0.5; LR 5e-4 → 1e-5 cosine with 10% warm-up; weight decay 1e-4; 200 epochs; rotflip augmentation; zero-shot hn panels; masking 0.6 / 8 px / ≤ 0.5 fully masked channels.
- 1 GPU: batch 4 × accumulation 2 = effective 8 (9128 optimizer steps per epoch). The `_2gpu` config differs only in accumulation 1, so a `last` checkpoint can move between the two when the second GPU frees up, as was done for V3-256.
- Sampler seed 42, the default. No torch seed is set, so initialisation is unseeded, as in every reference run.
- Checkpoints every 10 epochs (epoch_9, 19, ...), `from_checkpoint: last`.

Code: `multiplex_model/modules/equivariant/equivariant_modules_v2.py`. `pool_use_act=False` already existed in `EquivariantConvNeXtEncoder` (pool → PixelLN → blocks, no act) and applies to both the MA and the PM encoder. `use_hyperkernel_act` is new in `EquivariantMultiplexImageEncoder`: when false, `self.act` is `e2nn.IdentityModule`. Both keys reach the checkpoint's `model_config` through the permissive encoder config, so `run_validation.py` rebuilds the right architecture from the checkpoint. The branch `lukass/equiv-f256-noact` is `lukass/equiv-f256-fourier` plus this switch; `nonlinearity` stays at its default `norm`.

With the five activations gone the encoder has 528 fewer trainable parameters than V3-256 (one `log_bias` per field: 16 MA + 64 hyperkernel + 64 + 128 + 256 PM); nothing else in the state dict changes (checked in test b).

## Pre-launch tests (`tests_noact/test_noact_v3_256.py`)

CPU on bury (`ONLY=a,b,c`), GPU on szary (`sbatch scripts/szary/test_noact_v3_256.sh`, all four). Filled in before `sbatch` of the training job (CPU a,b,c on bury; GPU a,b job 9663, c job 9664, d job 9665); the reading below does not depend on them.

- **a** default flags rebuild V3-256 exactly (keys, shapes, values under seed 0): **passed** (bury, CPU, 2026-11-08). V3-256: 122.987 M trainable (encoder 33.505 M, decoder 89.482 M).
- **b** state dict of noact = V3-256 minus the five activation modules: **passed**. 15 entries removed (`log_bias`, `eps`, `indices_10` of `marker_agnostic_encoder.acts.0`, `encoder.act`, `pan_marker_encoder.acts.0/1/2`), 528 trainable parameters; no new keys; all shared shapes equal; decoder identical. `encoder.act` is `IdentityModule`; decoder acts unchanged (bottleneck `n_softplus`, 3 refine acts). noact: 122.986 M trainable.
- **c** (szary, GPU job 9664, fp32, TF32 off) D4 latent relative error of the random-init noact model: at most 9.6e-7 over all 8 elements, reconstruction at most 2.8e-7: **passed**. 45° reconstruction relative L2 in the central disk 0.168 (15°: 0.169; interpolation floor 0.047) vs 0.115 for V3-256 at random init; random-init values, not comparable with trained models. Stem output at random init: V3-256's stem NormAct zeroes 68.5% of the field vectors of a disc-masked input, the noact stem 16.4% (those outside the disc, where the zero-initialised bias gives an exactly zero fibre).
- **d** (szary, GPU job 9665; batch 4, 128 px input at 64 internal, 47 markers, bf16 train step, 1 RTX 5000 Ada): V3-256 69.0 s/100 steps, peak allocated 19.2 GiB; noact 68.2 s/100 steps, 19.7 GiB. Both project to about 3.5 h/epoch on one GPU, which is what the V3-256 job history shows (3.5 h on 1 GPU, 2.07 h on 2). No throughput difference. (V3-256 training logs: peak allocated 19.6–21.2 GiB, reserved 26–30 GiB.)

## Evaluation

Zero-shot leave-one-out on the 149 hn test patches with `run_validation.py`, exactly as for V3-256 (`scripts/szary/loo_hn_f256_noact.sh`, then `python scripts/szary/loo_aggregate.py <epoch> f256_noact`): 128 px, patch_mask_ratio 0, scored at 128, per-image `*_full.csv`, series in `logs/loo_f256_noact_series.txt`.

Epochs: e9 (collapse check), e29, e49, e79, then every 10 epochs if the run continues.

Reference, V3-256 (`logs/loo_f256_series.txt`), per-image Pearson / MSE:

| epoch | Pearson | pooled | MSE |
|---|---|---|---|
| e9 | .5546 | .6947 | .00848 |
| e29 | .6059 | .7465 | .00746 |
| e49 | .6128 | .7493 | .00693 |
| e79 | .6242 | .7624 | .00648 |
| e149 (best) | .6303 | .7684 | .00585 |
| e199 | .6311 | .7705 | .00573 |

V3-256 validation (training log, rank-0 shard): NLL −5.97 at e10, −6.37 at e24, −6.60 at e46, −6.70 at e69; MSE .00399 at e24, .00384 at e28.

## Pre-registered reading

- **Collapse check at e9.** Collapse is declared if the per-image LOO Pearson at e9 is below .40 (V3-256: .5546; a per-image constant prediction scores about 0) or if the reconstructions are visibly flat in the saved maps. If declared, training continues to e29 for confirmation and is then stopped; the result is recorded as a collapse and the primary is not evaluated.
- **Primary:** per-image LOO Pearson at e79, noact minus V3-256 (.6242). Same thresholds as the Fourier ablation, about half of the V3-256 vs V3-512 gap at e79:
  - **Gain** if ≥ +.004.
  - **Null** if within ±.004.
  - **Loss** if ≤ −.004.
  - The paired per-image z is reported as well, but the decision uses the threshold: single-run differences below .004 are within training-seed noise.
- **Secondary:** LOO MSE at e79 (V3-256 .00648) and validation NLL/MSE at matched epochs. A loss on the primary accompanied by a lower LOO MSE is reported as mixed, not as a loss.
- **Downstream (cell typing).** Because the confound concerns the representation and not the reconstruction, cell typing is run on the e79 checkpoint whatever the primary outcome (the protocol and epoch used for V3 in `FINAL_RESULTS_CELLTYPING_CLINICAL.md`), unless collapse was declared. A downstream difference to V3-256 e79 counts as real only if it exceeds half of the V3-256 vs V3-512 difference on the same metric and cohort. Clinical embeddings only if cell typing shows a real difference.
- **Training is stopped after e79** unless the primary is a gain or the LOO MSE is lower by 3% or more, in which case it runs to e199 like V3-256. Compute is the reason: V3-256 took about 2.1 h per epoch on two RTX 5000 Ada, roughly twice that on one.

No other modification is combined with this change in this run.
