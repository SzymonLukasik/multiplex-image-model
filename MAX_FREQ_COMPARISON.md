# `maximum_frequency` in ModelV2: What Changes Between 1, 2, and 3

This document explains exactly how `maximum_frequency` flows through your architecture, what each value buys you, and what it costs.

---

## Quick Reference Table

For your architecture (using `flipRot2dOnR2` = O(2)):

| `max_freq` | `repr_dim` per field | Irrep types in BL-regular | Hyperkernel `P/pair` (full) | Gspace cap needed | Angular resolution |
|------------|----------------------|-----------------------------|------------------------------|---------------------|---------------------|
| **1**      | 6                    | trivial(0,0)·1 + sign(1,0)·1 + freq₁(1,1)·2 | 6 (=2 + 4·1)                 | 2                   | Quadrant-level (90°) |
| **2**      | 10                   | + freq₂(1,2)·2              | 10 (=2 + 4·2)                | 4                   | Octant-level (45°)   |
| **3**      | 14                   | + freq₃(1,3)·2              | 14 (=2 + 4·3)                | 6                   | 16-sector (22.5°)    |

The general formula under O(2):
- **`repr_dim = 2 + 4 · max_freq`** (one trivial + one sign + 2 copies of each freq_k for k=1..M)
- **Channels per stage = `embedding_dims[stage] × repr_dim`**
- **Gspace cap = `2 · max_freq`** (needed for Clebsch-Gordan tensor products inside R2Conv)

---

## What `max_freq` Actually Controls

### 1. Angular resolution of the BL-regular fields

Each BL-regular field is a vector that transforms under rotation by a block-diagonal matrix:
- The trivial blocks don't change (invariant)
- The frequency-k block rotates by angle `kθ` (i.e., faster than the input rotation)

So `max_freq` sets the **highest angular harmonic** the model can natively represent at each pixel:

- **`max_freq=1`**: model can detect "edge orientations" but treats 0° and 180° patterns as different inputs only via the freq-1 sign-flip. Discretization errors at 45° and 135° are large.
- **`max_freq=2`**: model gains a "double-frequency" channel that distinguishes patterns separated by ~90°-period (e.g., crosshair vs. line at 45°).
- **`max_freq=3`**: triple-frequency channel — distinguishes patterns at 60°-period rotations.

For cellular morphology in IMC: cells are mostly isotropic (low frequency content) but membrane segments, fibrillar structures, and elongated cells have rotational structure that benefits from at least freq-1, possibly freq-2.

### 2. Per-pixel channel count

Every BL-regular field expands to `repr_dim` channels in the underlying tensor. Your stage embedding dims are *number of fields*, not raw channels:

| Stage example: `pm_embedding_dims: [128, 256, 512]` | max_freq=1 | max_freq=2 | max_freq=3 |
|---|---|---|---|
| Stage 0 channels: `128 × repr_dim` | 768 | 1280 | 1792 |
| Stage 1 channels: `256 × repr_dim` | 1536 | 2560 | 3584 |
| Stage 2 channels: `512 × repr_dim` | 3072 | 5120 | 7168 |

Memory and compute scale linearly with channel count (so ~2.3× from freq-1 → freq-3).

### 3. ESCNN's irrep cache requirement

This is a subtle implementation point: ESCNN's R2Conv builds kernel bases by computing **Clebsch-Gordan decompositions of tensor products** of irreps in your BL-regular field. The product `freq_k ⊗ freq_k` produces frequencies up to `2k`, so ESCNN needs irreps up to `2·max_freq` to be in its cache.

Your code now enforces this:
```python
self._gspace = escnn.gspaces.flipRot2dOnR2(N=-1, maximum_frequency=2 * maximum_frequency)
```

This is why you hit `InsufficientIrrepsException` before — the gspace cap was too low. With this fix, all max_freq values work.

---

## How Each max_freq Affects Each Stage of Your Architecture

### Stage 1: Marker-Agnostic (MA) Encoder

**Operates on**: single-channel marker images, lifted to BL-regular fields.

**Impact of max_freq**:
- Higher `max_freq` → more angular detail captured per marker per pixel
- For your config `ma_embedding_dims: [16]`: stage output has `16 × repr_dim` channels
  - max_freq=1: 96 channels per marker per pixel
  - max_freq=2: 160 channels
  - max_freq=3: 224 channels
- Cost scales linearly with `repr_dim`; this stage is small so the absolute cost is low.

**Why it matters**: this is where local cellular morphology (membranes, organelles, edges) is encoded. If your markers stain oriented structures (vessels, fiber tracts), higher max_freq captures their orientation natively rather than via spatial gradients.

### Stage 2: EquivariantHyperkernel (Marker Mixing)

**Operates on**: stacked per-marker fields, learns marker-conditioned mixing.

**Impact of max_freq depends on `intertwiner_basis`**:

#### `intertwiner_basis: scalar` (`P = 1`)
Hyperkernel parameter count is **independent of max_freq**:
- `params = num_channels × K_in × K_out × 1`
- Always cheap, regardless of `max_freq`

#### `intertwiner_basis: full` (`P = 2 + 4·max_freq`)
Hyperkernel scales with max_freq:
| max_freq | P/pair | Hyperkernel params (your config: `num_channels=60, K_in=16, K_out=384`) |
|---|---|---|
| 1 | 6  | 60 × 16 × 384 × 6  = 2.21 M |
| 2 | 10 | 60 × 16 × 384 × 10 = 3.69 M |
| 3 | 14 | 60 × 16 × 384 × 14 = 5.16 M |

The "full" basis adds expressivity at higher max_freq because it can mix the multiplicity copies of each frequency irrep separately. Without it, the multiplicity copies are forced to scale uniformly — wasting capacity.

**Recommendation**: if you increase `max_freq`, also use `intertwiner_basis: full` to actually exploit the added angular resolution. Otherwise the higher-frequency irreps just propagate through with a single shared scale per field, which gives them less learned flexibility than they deserve.

### Stage 3: Pan-Marker (PM) Encoder

**Operates on**: the BL-regular feature map produced by the hyperkernel, applies 3 stages of equivariant ConvNeXt.

**Impact of max_freq is largest here** because:
- Each `BLConvNeXtBlock` contains depthwise + pointwise R2Conv layers, all of which scale with `repr_dim`
- The depthwise R2Conv basis size grows as ~`repr_dim²` (since basis is over input × output irrep pairs)
- ESCNN's basis-expansion cost at construction time grows roughly as `O(max_freq³)` due to Clebsch-Gordan decompositions

**For your config `pm_embedding_dims: [128, 256, 512]`**: the channel count entering the final stage is `512 × repr_dim`:
- max_freq=1: 3072 channels
- max_freq=2: 5120 channels
- max_freq=3: 7168 channels

**Wall-clock impact**: training-time forward pass should be roughly `repr_dim`× slower per stage. So freq-3 vs freq-1 is roughly 2.3× slower per BL block.

### Stage 4: Regular2Trivial Projection

**Operates on**: final BL-regular tensor → `output_scalars` trivial scalars.

This is a 1×1 R2Conv that **picks the trivial-rep combination** from the rich field representation. It uses `Σ K_fields × output_scalars × intertwiner_dim_for_trivial` parameters.

For BL-regular → trivial, the intertwiner space is exactly the trivial sub-component of the input field — i.e., the projection learns weights that combine the trivial(0,0) and (potentially) sign(1,0) channels of each input field into the output scalars.

**Why this matters for max_freq**: even though the projection itself is small, the **information** that ends up in the trivial scalars depends on what the encoder did to the higher-frequency components beforehand. Higher max_freq lets the model compute richer non-trivial features that then get *combined* into invariant descriptors via Clebsch-Gordan in earlier R2Convs. So max_freq affects what the codebook can express, even though the final projection is dimension-cheap.

### Stage 5: Decoder (Non-Equivariant)

**No impact**: the decoder operates on the trivial latent and is identical regardless of max_freq.

---

## Practical Trade-offs

### Memory

| max_freq | Approx. peak GPU memory (relative) | Reason |
|----------|--------------------------------------|--------|
| 1 | 1.0× (baseline) | repr_dim=6 |
| 2 | ~1.7× | repr_dim=10, ~67% more channels |
| 3 | ~2.3× | repr_dim=14, ~133% more channels |

### Wall-clock time per epoch

| max_freq | Approx. time (relative) | Reason |
|----------|---------------------------|--------|
| 1 | 1.0× | baseline |
| 2 | ~1.5–2× | proportional to channel count + ESCNN basis costs |
| 3 | ~2.5–4× | bigger ESCNN basis, slower init, slower per-batch |

### Construction time at startup

ESCNN computes Clebsch-Gordan decompositions and kernel bases at module construction. This is one-time cost but can be slow:
- max_freq=1: <30 seconds
- max_freq=2: ~1–2 minutes
- max_freq=3: ~3–5 minutes (may even longer the first run; cached afterwards)

### Equivariance error reduction

In theory, higher max_freq should lower equivariance error under continuous rotations because:
- Higher angular harmonics in the basis better approximate rotation-equivariant kernels on the discrete grid
- The "aliasing floor" of the discrete grid limits how much improvement you can get; eventually max_freq hits diminishing returns

In practice for IMC-style data, expect:
- **max_freq=1 → 2**: noticeable improvement (from quadrant-level to octant-level resolution)
- **max_freq=2 → 3**: smaller improvement (most of the angular detail is already captured)
- **max_freq > 3**: typically diminishing returns; not worth the cost

---

## How max_freq Interacts with Other Choices in Your Model

### With `antialiased_downsample`

Higher max_freq is **only beneficial** if the downsampling preserves the higher-frequency content. With `antialiased_downsample: false`, stride-2 R2Convs alias the higher frequencies aggressively — so max_freq=3 with antialiasing off may be no better than max_freq=1.

**Recommendation**: always pair higher max_freq with `antialiased_downsample: true`.

### With `output_scalars` and `decoder.decoded_embed_dim`

The final `Regular2Trivial` learns a projection from `pm_embedding_dims[-1] × repr_dim` channels to `output_scalars` trivial channels. The codebook size (`output_scalars`) should generally scale with the richness of the upstream representation:
- max_freq=1, 512 fields → 3072 input channels → `output_scalars=512` is reasonable
- max_freq=2, 512 fields → 5120 input channels → `output_scalars=512` is still fine but you might benefit from 768
- max_freq=3, 512 fields → 7168 input channels → consider `output_scalars=1024` to avoid over-compression

### With `intertwiner_basis`

| | scalar | full |
|---|---|---|
| max_freq=1 | OK baseline | minor improvement (6 vs 1 weight per pair) |
| max_freq=2 | wastes the higher-freq capacity | gives independent control over freq_2 mixing |
| max_freq=3 | likely under-utilized | full payoff: separate control per frequency |

**Rule of thumb**: if you're paying the cost of higher max_freq, also pay for `intertwiner_basis: full` so the hyperkernel can actually exploit it.

### With `use_norm` and `EquivariantPixelLN`

`EquivariantPixelLN` operates per-field. Higher max_freq → larger `repr_dim` → more components per field → the per-field L2 norm captures a richer notion of "field magnitude". Norm normalization should help training stability more at higher max_freq, where un-normalized higher-freq components could blow up.

**Recommendation**: keep `use_norm: true` for max_freq ≥ 2.

---

## My Recommendations for Your Setup

Based on:
- Your config (`pm_embedding_dims: [128, 256, 512]`, `output_scalars: 512`, `antialiased_downsample: true`)
- Your goal (latent equivariance + reconstruction quality on IMC virtual staining)
- Your reported v2 results (perfect discrete equivariance, solid continuous equivariance at max_freq=1)

### **Most-likely-best choice: max_freq=2 with `intertwiner_basis: full`**

This gives you:
- ~1.7× memory and ~1.5× compute vs. max_freq=1
- Octant-level (45°) angular resolution — meaningful improvement for continuous rotations
- Full intertwiner so the freq_2 components are actually exploited
- Hyperkernel param count grows from 0.37M → 3.7M (modest in the context of total ~10M+ params)

### **Worth trying: max_freq=3 with `intertwiner_basis: full`**

If max_freq=2 shows clear improvement and you have GPU budget:
- ~2.3× memory, ~2.5× compute
- 16-sector (22.5°) angular resolution — fine but probably overkill for cellular morphology
- Diminishing returns; only worth it if max_freq=2 results suggest more angular resolution would still help

### **Likely not worth it: max_freq=1 with `intertwiner_basis: full`**

The "full" basis at max_freq=1 only adds 6× hyperkernel parameters but doesn't give you more angular resolution. If your max_freq=1 model with scalar basis already works well, sticking with scalar at max_freq=1 is more parameter-efficient.

### **Decision tree**

```
Have you measured equivariance error vs. rotation angle?
├── No → measure it first. Plot error vs. angle for max_freq=1 baseline.
└── Yes → Is the error spike sharpest at angles like 22.5°, 45°, 67.5°?
    ├── Yes → max_freq=2 should help (octant resolution covers 45°)
    └── No → max_freq=1 is probably enough; spend budget elsewhere
        (more pm_embedding_dims, more pm_layers_blocks, larger output_scalars)
```

---

## TL;DR

`maximum_frequency` controls the **angular resolution** of the BL-regular field representation:
- **Higher values = more orientation-sensitive features per pixel**, at proportional memory/compute cost
- For O(2) (`flipRot2dOnR2`), the field size grows as `2 + 4·max_freq`
- The gspace must be constructed with cap = `2·max_freq` (already handled by your code)
- Pair with `antialiased_downsample: true` and `intertwiner_basis: full` to actually exploit the added expressivity
- Sweet spot for IMC-like data: **max_freq=2 with `intertwiner_basis: full`**
