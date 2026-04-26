# ESCNN Insufficient Irreps Error Analysis

## Error Summary

```
escnn.group._numerical.InsufficientIrrepsException: 
Error! Did not find sufficient irreps to complete the decomposition of the tensor product of 'irrep_1,2' and 'irrep_1,2'.
It is likely this happened because not sufficiently many irreps in 'O(2)' have been instantiated.
Try instantiating more irreps and then repeat this call.
The sum of the sizes of the irreps found is 2, but the representation has size 4.
```

## Root Cause

**ESCNN's Clebsch-Gordan solver cannot decompose the tensor product of two frequency-2 irreps under O(2)** because it hasn't cached/instantiated all necessary irreps.

The error occurs at model instantiation when constructing `BLConvNeXtBlock`'s depthwise convolution. When ESCNN tries to build the kernel basis for the R2Conv, it needs to compute Clebsch-Gordan coefficients for decomposing tensor products of the BL-regular representation's irreps. For `maximum_frequency=3`:

- **BL-regular irreps**: trivial (0) + frequency 1 + frequency 2 + frequency 3
- **Tensor product attempt**: frequency 2 ⊗ frequency 2 → frequencies {0, 1, 2, 3, 4}
- **Problem**: ESCNN was only initialized with irreps up to frequency 3, but the tensor product requires frequency 4 to decompose correctly.

The error message "The sum of the sizes of the irreps found is 2, but the representation has size 4" means ESCNN found only trivial (size 1) and frequency-1 (size 2), totaling 3, but needs 4+ to represent the full tensor product.

## Why It Happens with Your Config

Your config in the training job had:

```yaml
encoder:
  pm_embedding_dims: [64, 128, 256]
  maximum_frequency: 3
```

The issue **is not** with `output_scalars` or the antialiased downsampling—it's purely an ESCNN limitation with `maximum_frequency=3`. When you construct an O(2) group with `maximum_frequency=3`, ESCNN should theoretically handle tensor products of those irreps, but in practice the Wigner-Eckart solver caches irreps on demand, and frequency-4 isn't in the cache.

## Solutions

### Option 1: Use `maximum_frequency=2` (Recommended for Quick Test)

```yaml
encoder:
  pm_embedding_dims: [64, 128, 256]
  maximum_frequency: 2  # repr_dim = 1 + 2*2 = 5
  output_scalars: 512
```

**Why it works**: Frequency 2 ⊗ Frequency 2 = {0, 1, 2, 3, 4}, but ESCNN will instantiate up to frequency 4 as a side effect. The cache handles it.

**Trade-off**: Less angular resolution than frequency 3, but faster convergence and lower VRAM.

### Option 2: Manually Pre-instantiate Higher Irreps (Advanced)

Add to your training script before model construction:

```python
import escnn.group as group_module
# Force instantiation of higher frequency irreps
g = escnn.gspaces.flipRot2dOnR2(N=-1, maximum_frequency=4)
_ = g.fibergroup.bl_regular_representation(3)
```

Then use `maximum_frequency=3` in config. This pre-caches the needed irreps before ESCNN tries to use them.

### Option 3: Update ESCNN

Check if a newer version of ESCNN/e2cnn fixes this. The caching behavior has improved in recent versions. Your venv uses an older ESCNN installation—upgrading might help, but test locally first.

## Recommended Action

**Start with `maximum_frequency: 2`** in your config:

```yaml
encoder:
  pm_embedding_dims: [64, 128, 256]
  maximum_frequency: 2
  output_scalars: 512
```

This avoids the tensor-product decomposition issue entirely while still giving you 5-dimensional BL-regular fields (vs. 3 for frequency 1). If you later want frequency 3, implement Option 2 in the training script.

## Files Affected

- `train_masked_equivariant_config_flip_v2_wider_modelv2.yaml` — change `maximum_frequency: 3` → `maximum_frequency: 2`

---

**References**:
- ESCNN WignerEckartBasis solver (joblib-cached tensor product decomposition)
- O(2) group irreps: trivial, frequency-k pairs (2D irreps) for k ≥ 1
- BL-regular representation: direct sum of trivial + all frequency-k up to max_freq
