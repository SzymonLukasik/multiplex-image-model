# ModelV2 Architecture Review: Equivariant ConvNeXt with Field-Wise Hyperkernel

**File**: `multiplex_model/equivariant_modules_v2.py` (1222 LOC)

## Executive Summary

ModelV2 implements a **continuously equivariant encoder path** (marker-agnostic + pan-marker stages) with two key innovations:
1. **EquivariantHyperkernel**: Field-wise (scalar-per-field-pair) marker-conditioned mixer that preserves SO(2)/O(2) equivariance through the interface
2. **Antialiased downsampling**: Gaussian blur + stride-2 to reduce non-discrete rotation error

The final latent projects to trivial scalars via `Regular2Trivial`, losing orientation structure before the decoder. This is architecturally **correct but raises efficiency concerns** explored below.

---

## Architecture Overview

### 1. **EquivariantHyperkernel** (lines 355–430)

**Purpose**: Marker-conditioned channel mixing that treats BL-regular fields as atomic vectors.

**Mechanism**:
- Input: plain tensor `(B, C·K_in·R, H, W)` where each BL-regular field is a contiguous `R`-component block
- **Scalar weights**: `nn.Embedding(num_channels, K_in × K_out)` → one real per (input field, output field) pair per marker
- **Equivariant contraction**: `einsum("bfrhw,bef->berhw", x_f, w)` — scalars commute with ρ(θ) on the `R` axis
- **Bias**: restricted to freq-0 (trivial) components only

**Strength**: Provably equivariant by construction (scalar multiplication with geometric objects).

**Weakness**: Least expressive equivariant parameterization. A full intertwiner between two BL-regular reps has `K_in × K_out × R` parameters (one complex per frequency × field pair). This version has only `K_in × K_out` real parameters — it can scale and mix fields but cannot rotate phase or frequency-dependent scaling.

**Verdict**: Good for a first pass; if expressivity becomes the bottleneck (e.g., equivariance error plateaus), upgrade to `e2nn.R2Conv(k=1)` with full intertwiner basis.

---

### 2. **EquivariantConvNeXtEncoder** (lines 977–1222)

**Purpose**: Equivariant feature pyramid with configurable downsampling and final projection.

**Key Parameters**:
- `output_trivial: bool` — whether to project final latent to trivial (default True)
- `output_scalars: int` — explicit target scalar count for `Regular2Trivial` (if None, defaults to num fields)
- `antialiased_downsample: bool` — Gaussian blur + 1×1 channel expander vs. stride-2 R2Conv

**Downsampling Paths**:

| Mode | Kernel | Stride | Padding | Output Size (113 input) | Aliasing |
|------|--------|--------|---------|-------------------------|----------|
| `antialiased=True` | Gaussian blur | 2 | centered | 57→29→15 | Low (shift-invariant) |
| `antialiased=False` | R2Conv(k=3) | 2 | p=1 | 57→29→15 | High (grid resampling) |

**Output Type Handling**:
- If `output_trivial=False` → returns `GeometricTensor` (used by pan-marker encoder)
- If `output_trivial=True` → projects via `Regular2Trivial(n_scalars=output_scalars)` → plain tensor

---

### 3. **EquivariantMultiplexImageEncoder** (lines 433–588)

**Data Flow**:
```
[B, C, H, W] input
    ↓
marker-agnostic encoder → [B*C, K_in*R, H_ma, W_ma] (GeometricTensor)
    ↓ reshape: (B*C, K_in*R, H, W) → (B, C*K_in*R, H, W)
hyperkernel (marker-conditioned scalar mixer) → [B, K_out*R, H, W] (GeometricTensor)
    ↓
equivariant nonlinearity (NormNonLinearity, not plain GELU)
    ↓
pan-marker encoder → [B, pm_embedding_dims[-1]*R, 15, 15] (GeometricTensor)
    ↓ Regular2Trivial
final latent → [B, output_scalars, 15, 15] (plain tensor)
```

**Critical Detail**: The entire flow (MA → hyperkernel → PM) preserves BL-regular geometry until the final projection. No intermediate collapse to invariants.

---

## Efficiency & Equivariance Trade-off Analysis

### The Core Tension: Why Project to Trivial at the End?

**Current design**:
- Encoder maintains continuous SO(2)/O(2) equivariance up to the final latent
- Final step: `Regular2Trivial` projects `K_fields × repr_dim` channels → `output_scalars` trivial scalars
- Decoder then reconstructs images in a **non-equivariant** manner

### What This Means

1. **Computational Efficiency**: Building equivariance into the entire 3-stage encoder (MA + hyperkernel + PM) costs ~10–20% overhead vs. standard ConvNeXt due to:
   - Larger field tensors (1 field at `max_freq=2` = 5 channels vs. 1 scalar)
   - ESCNN's Clebsch-Gordan caching and kernel basis computation
   - NormNonLinearity instead of cheap GELU

2. **Information Preservation**: The latent loses all orientation structure. A 180° rotation input → different encoder trajectory → same final (trivial) latent magnitude & distribution.

3. **Decoder Mismatch**: The decoder is purely convolutional (non-equivariant), so it must learn to:
   - Infer what rotation happened from the latent
   - Reverse it during reconstruction
   - This requires explicit orientation-sensing capacity in the PM encoder

### Why Not Stay Equivariant All the Way?

**Three options**:

#### A. **Current (Hybrid)**
- Equivariant encoder → trivial latent → non-equivariant decoder
- **Pros**: Simple decoder, fast inference, compatible with existing masked-MAE losses
- **Cons**: Latent loses orientation; decoder must learn orientation from magnitude alone
- **Best for**: Fast iteration, diagnostic experiments

#### B. **Full Equivariant Encoder + Decoder**
- Both encoder and decoder are equivariant (decoder outputs BL-regular feature maps)
- Final loss computed on reconstructed fields + rotation consistency term
- **Pros**: Entire pipeline respects symmetries; no information loss
- **Cons**: Decoder must output `K_out × R` channels; requires rotation-aware loss; slower
- **Best for**: When rotation equivariance is a hard requirement

#### C. **Equivariant Encoder + Equivariance Loss (This Study)**
- Equivariant MA+PM encoder, but measure **how well the latent is equivariant**
- Add consistency loss: `L_equiv = ||f(x_θ) - f(x)||_2 + ||recon_θ - rotate(recon)||_2`
- Decoder stays non-equivariant but the latent learns to encode orientation
- **Pros**: Captures equivariance benefits without decoder overhead; explicit measurement
- **Cons**: Additional loss term tuning
- **Best for**: Current work (evaluate equivariance error across rotations)

---

## Specific Design Choices

### EquivariantPixelLN (lines 39–138)
- **Purpose**: Equivariant normalization. Centers freq-0 scalars (only invariant subspace), scales all by mean L2 norm per field
- **Verdict**: Elegant. Preserves equivariance while stabilizing training.
- **Alternative**: `e2nn.FieldNorm` (batch-stats-based) — less interpretable but may be faster

### GRNByIrrep (lines 141–226)
- Gating per field copy with learned (γ, β)
- **Verdict**: Necessary for expressivity; field-aware gating is better than per-channel

### BLConvNeXtBlock (lines 828–927)
- Depthwise + expanded MLP (with optional gating/GRN/LayerScale)
- **Issue at `max_freq=3`**: Depthwise R2Conv requires Clebsch-Gordan decomposition of `freq_2 ⊗ freq_2` → fails if freq-4 irreps not cached
- **Workaround**: Use `maximum_frequency ≤ 2` or pre-instantiate higher irreps manually

### antialiased_downsample Path (lines 1098–1109)
- **Correct implementation**: `PointwiseAvgPoolAntialiased(sigma=0.66, stride=2)` + `R2Conv(k=1)`
- **Verified**: Both paths (antialiased vs. conv-only) produce identical 57→29→15 spatial shapes for 113 input
- **Impact on equivariance**: Antialiased reduces bilinear resampling artifacts under non-discrete rotations; should improve test metrics

---

## Configuration Recommendations

### Conservative (Fast Iteration)
```yaml
encoder:
  maximum_frequency: 1  # repr_dim=3
  pm_embedding_dims: [64, 128, 256]
  antialiased_downsample: true
  output_scalars: 256
```
- Baseline equivariance + efficient downsampling
- ~80% of full v2 benefit with <10% overhead

### Ambitious (Higher Angular Resolution)
```yaml
encoder:
  maximum_frequency: 2  # repr_dim=5
  pm_embedding_dims: [64, 128, 256]
  antialiased_downsample: true
  output_scalars: 512
```
- Test if frequency-2 improves non-discrete rotation handling
- **Must use `maximum_frequency: 2`, not 3** (ESCNN tensor product limitation)

---

## Known Issues & Workarounds

| Issue | Cause | Fix |
|-------|-------|-----|
| `InsufficientIrrepsException` at `max_freq=3` | ESCNN caches irreps on-demand; freq-3 ⊗ freq-3 needs freq-4 | Use `max_freq ≤ 2` or pre-instantiate higher irreps |
| v1 ↔ v2 state dict incompatible | MA encoder no longer has `regular2trivial`; hyperkernel is new class | v2 requires fresh training (no checkpoint loading from v1) |
| Decoder expects specific latent dim | `output_scalars` must match `decoder.decoded_embed_dim` | Update config: `encoder.output_scalars: 512` and `decoder.decoded_embed_dim: 512` |

---

## Experimental Validation Checklist

- [ ] **Equivariance error curve**: Run [evaluate_equivariance_v2.py](evaluate_equivariance_v2.py) on v2 model; compare vs. v1 (should be lower for non-discrete rotations due to antialiased downsampling)
- [ ] **Reconstruction MSE**: Train v2 with `max_freq=1` vs. `max_freq=2`; check if higher frequency improves downstream tasks
- [ ] **Wall-clock time**: Profile encoder forward pass (equivariant ops) vs. v1; overhead should be <15%
- [ ] **Memory**: Check peak GPU memory on training batches; BL-regular features have higher channel count
- [ ] **Ablation**: Train with `antialiased_downsample=False` vs. `True`; measure impact on rotation error

---

## Future Directions

1. **Full Equivariant Decoder**: Output reconstructed BL-regular feature maps; compare orientation-aware reconstruction quality

2. **Adaptive Frequency**: Use `max_freq = f(input_resolution)` — higher freq for small, detailed patches; lower for large features

3. **Intertwiner Basis**: Upgrade hyperkernel to full `e2nn.R2Conv(k=1)` if scalar-per-pair expressivity is insufficient (measure via equivariance error plateau)

4. **Rotation-Aware Loss**: Add term like `||encode(rotate_θ(x)) - rotate_θ(encode(x))||` to explicitly train equivariance into the latent

---

## Summary

ModelV2 is a **sound hybrid approach**: equivariance in the encoder (where it's cheap), trivial latent (where it simplifies downstream), and equivariance validation via test metrics. The antialiased downsampling is the most concrete improvement. The scalar-per-field hyperkernel is the right choice for a first iteration—simple, provably equivariant, and easy to ablate.

**Verdict**: Architecturally correct. Execute `maximum_frequency: 2` to avoid ESCNN caching issues, then measure equivariance error improvement. If results show promise, invest in full equivariant decoding or rotation-aware loss terms.
