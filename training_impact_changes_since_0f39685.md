# Training-impacting changes since `0f396853fd2c3867e4b1565fb745a4c0a6674944`

This note focuses only on changes that *practically alter model training behavior* (optimization dynamics, model capacity, masking objective, evaluation signals), rather than repository structure alone.

## 0) Baseline (`0f39685`, Jan 12) in one line
The Jan 12 `update modules` commit was a targeted update in the old monolithic `multiplex_model/modules.py`; the later refactor period changed the training pipeline much more deeply (config validation, modular backbones, masking/loss flow, logging/eval metrics).

---

## 1) Latent normalization is now defaulted on (direct effect on latent scale/stability)
**Practical impact:** latent activations entering the decoder are now normalized by default, which usually stabilizes optimization and can change reconstruction quality/calibration.

- Config default is now `use_latent_norm=True` in encoder config.
- The encoder applies `LayerNorm` on latent only if enabled.

```python
# multiplex_model/utils/configuration.py
use_latent_norm: bool = Field(default=True)
```

```python
# multiplex_model/modules/immuvis.py
self.latent_norm = (
    LayerNorm(pm_embedding_dims[-1], data_format="channels_first")
    if use_latent_norm
    else nn.Identity()
)
...
x = self.latent_norm(x)
```

Why this matters for training:
- Normalized latent distributions reduce scale drift across batches/panels.
- Interacts with AdamW + cosine schedule by making effective step sizes less erratic.

---

## 2) Encoder/decoder architecture is now registry-driven (backbone choice affects convergence/capacity)
**Practical impact:** training can now swap ConvNeXt / ViT / Swin / ResNet style components by config rather than code edits. This changes inductive bias, parameterization, and GPU/memory profile.

- Registry-based resolution selects block/encoder classes from config.
- Package includes dedicated implementations for ConvNeXt, ViT, Swin, ResNet (+ multiplex-specific components).

```python
# multiplex_model/modules/registry.py
cls = registry.get(config["type"])
module_parameters = config.get("module_parameters", {})
return cls(**module_parameters)
```

```python
# multiplex_model/modules/immuvis.py
block_cls = resolve_block_class(block_type)
...
self.decoder = nn.Sequential(*[block_cls(decoded_embed_dim, **block_kwargs) for _ in range(num_blocks)])
```

Why this matters for training:
- Backbone swap changes optimization landscape and required hyperparameters.
- Enables controlled experiments without code drift (fewer accidental implementation differences).

---

## 3) Masking objective is explicitly two-stage in train, one-stage in validation
**Practical impact:** train-time corruption is stronger and more diverse than val-time corruption, which materially changes learned invariances.

Train loop behavior:
1. Optional random channel subset sampling.
2. Full channel dropping on a subset.
3. Spatial patch masking on remaining active channels.

Validation behavior:
- No channel subset sampling, only full channel masking + spatial masking.

```python
# train_masked_model.py (train)
img, channel_ids, masked_img, active_channel_ids = apply_channel_masking(..., apply_channel_subset_sampling=True)
masked_img, _ = apply_spatial_masking(masked_img, spatial_masking_ratio, mask_patch_size)
```

```python
# train_masked_model.py (val)
_, _, masked_img, active_channel_ids = apply_channel_masking(..., apply_channel_subset_sampling=False)
masked_img, pixel_mask = apply_spatial_masking(masked_img, spatial_masking_ratio, mask_patch_size)
```

```python
# multiplex_model/utils/masking.py
num_sampled_channels = np.random.randint(min_channels, num_channels + 1)
...
num_channels_to_mask = np.random.randint(1, max_channels_to_mask + 1)
```

Why this matters for training:
- Model is forced to reconstruct from both missing channels and missing pixels.
- Changes effective task difficulty and signal-to-noise ratio per step.

---

## 4) Uncertainty-aware objective path is explicit (`mi`, `logvar`, beta-NLL) with clamped log-variance gradients
**Practical impact:** model optimizes a probabilistic target (mean + variance) instead of only point estimates; this alters both gradients and calibration behavior.

```python
# train_masked_model.py
output = model(masked_img, active_channel_ids, channel_ids)["output"]
mi, logvar = output.unbind(dim=-1)
mi = torch.sigmoid(mi)
logvar = ClampWithGrad.apply(logvar, -15.0, 15.0)
loss = beta_nll_loss(img, mi, logvar, beta=beta)
```

```python
# multiplex_model/utils/optim.py
class ClampWithGrad(torch.autograd.Function):
    ...
    return x.clamp(min_val, max_val)
```

Why this matters for training:
- Predictive variance is learned jointly with reconstruction mean.
- Clamping avoids unstable variance extremes while retaining smooth gradients outside clamp bounds.

---

## 5) Optimization mechanics tightened: AMP+bfloat16, grad accumulation, grad clipping, warmup+cosine annealing
**Practical impact:** effective batch dynamics and LR trajectory are now more controlled and scalable.

```python
# train_masked_model.py
with autocast(device_type="cuda", dtype=torch.bfloat16):
    ...
scaler.scale(loss / gradient_accumulation_steps).backward()
...
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scheduler.step()
```

```python
# multiplex_model/utils/optim.py
if current_step < num_warmup_steps: ...
...
return final_lr_mult + (1.0 - final_lr_mult) * 0.5 * (1.0 + cos(pi * progress))
```

Why this matters for training:
- Mixed precision improves throughput; gradient scaling protects low-precision stability.
- Warmup + cosine decay often reduces early divergence and improves late-stage fine-tuning.

---

## 6) Evaluation signals are richer: latent rank + variance/MAE correlation
**Practical impact:** validation now tracks representation quality and uncertainty usefulness, not just reconstruction error.

```python
# train_masked_model.py
rankme = RankMe(all_latents)
variance_mae_corr = torch.corrcoef(
    torch.stack([all_channel_variances.flatten(), all_channel_maes.flatten()])
)[0, 1].item()
```

Why this matters for training decisions:
- `latent_rankme` can reveal collapse/overcompression trends.
- Variance-vs-error correlation indicates whether uncertainty estimates are informative.

---

## 7) Config validation is stricter (fewer silent bad runs)
**Practical impact:** many invalid settings now fail fast at startup instead of producing misleading training runs.

Examples:
- Positive checks for block counts and embedding dims.
- Enforced length match between `*_layers_blocks` and `*_embedding_dims`.
- Pan-marker layers cannot be empty.

```python
# multiplex_model/utils/configuration.py
if len(v) != len(blocks):
    raise ValueError(...)
...
if len(v) == 0:
    raise ValueError("pm_embedding_dims cannot be empty ...")
```

Why this matters for training:
- Less wasted GPU time on malformed experiments.
- Better reproducibility across runs/config files.

---

## Quick takeaway
The most consequential practical shifts are: **(a)** default latent norm, **(b)** modular backbone selection, **(c)** stronger/more explicit masking curriculum, **(d)** uncertainty-aware beta-NLL training with stabilized log-variance, and **(e)** more controlled optimization + richer validation diagnostics.
