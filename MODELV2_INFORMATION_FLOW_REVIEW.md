# ModelV2: Information Flow, Equivariance, and the Trivial Projection

## 1. Information Flow Through the Model

Your model has three semantically distinct stages, each operating on a different "view" of the data. Understanding what flows where is the key to understanding why this architecture is sensible.

### Stage 1: Marker-Agnostic (MA) Encoder

**Operates per-channel, per-marker independently.**

```
[B, C, H, W] input (C markers, treated independently)
    ↓ reshape to [B*C, 1, H, W]
MA encoder (BL-regular fields, equivariant)
    ↓
[B*C, K_in*R, H_ma, W_ma] GeometricTensor
```

**What flows here**: marker-agnostic morphological features. The encoder learns *what cellular structures look like* without knowing which marker is being viewed. Each marker channel produces its own feature map, but the *parameters are shared* across markers.

**Why equivariance matters here**: if the model sees CD3 staining of T-cells, the morphological features should not depend on the orientation of the slide. A T-cell at 30° rotation should produce the same channel-wise descriptors as one at 0°. The BL-regular fields make this structurally true.

**Information content**: high-dimensional per-marker feature maps. Each spatial position carries `K_in × R` channels (e.g., 16 fields × 5 components = 80 channels at `max_freq=2`). The frequency components encode orientation locally — a directed edge produces non-zero energy in the freq-1 components, isotropic blobs produce only freq-0.

### Stage 2: EquivariantHyperkernel — Marker Mixing

**Operates per-pixel, mixing across markers.**

```
[B*C, K_in*R, H_ma, W_ma]
    ↓ reshape: collapse C into the field axis
[B, C*K_in*R, H_ma, W_ma]
    ↓ EquivariantHyperkernel: marker-conditioned scalar mixing
[B, K_out*R, H_ma, W_ma]
```

**What flows here**: a *contextualized* multi-marker representation. For each pixel, the hyperkernel asks: "given this combination of markers (CD3 + DAPI + Vimentin) at this location, what is the joint cellular descriptor?"

**Why this is the conceptual centerpiece**: this is where marker identity actually enters the computation. The MA encoder doesn't know which marker is which — it just produces generic features. The hyperkernel says "given marker tokens [`CD3_idx, DAPI_idx, Vimentin_idx`], here's how to combine the per-marker features into a single multi-marker vector at each pixel."

**The equivariance constraint**: the mixing weights are scalars (one per (input field, output field, marker) triplet). Scalars commute with rotation matrices acting on the field components, so the entire mixing operation respects rotational symmetry. **This is non-trivial** — a naive learned mixing layer would not be equivariant.

### Stage 3: Pan-Marker (PM) Encoder

**Operates spatially on the mixed multi-marker representation.**

```
[B, K_out*R, H_ma, W_ma]
    ↓ pan-marker encoder (3 stages of BL-regular ConvNeXt)
[B, pm_emb[-1]*R, 15, 15] GeometricTensor
    ↓ Regular2Trivial projection
[B, output_scalars, 15, 15] plain tensor (the latent)
```

**What flows here**: progressively coarsened spatial features representing tissue-level patterns. The receptive field grows; local cellular descriptors aggregate into local-context descriptors (e.g., "T-cell surrounded by stroma" vs. "T-cell surrounded by other T-cells").

**Why equivariance matters here too**: the spatial aggregation should not depend on tissue orientation. A vessel running diagonally should be detected with the same confidence as one running horizontally. This is the classical CNN equivariance argument.

---

## 2. Does Equivariance Give You More Expressiveness?

**Short answer**: No, equivariance constrains the function space — it gives you *less* expressiveness for the same parameter count. But it gives you *better generalization* per parameter, *more sample efficiency*, and *structural guarantees*.

### The expressiveness math

Consider a linear layer mapping between two BL-regular field types of dimension `n × R` and `m × R`:

| Parameterization | Free parameters | Expressiveness |
|---|---|---|
| Unrestricted linear | `(nR) × (mR) = nmR²` | Maximum, but breaks equivariance |
| Full intertwiner basis | `nm × R` (1 real for trivial + 1 complex per freq) | Maximal *equivariant* expressiveness |
| Scalar-per-field-pair (your hyperkernel) | `nm` | Strict subset of intertwiner basis |
| Diagonal scalar | `min(n,m)` | Just rescaling, no mixing |

For `R=5` (max_freq=2): full intertwiner is `5×` more expressive than scalar-per-pair, and unrestricted is `25×` more expressive than full intertwiner.

### What equivariance actually does

It **trades expressiveness for inductive bias**. Three concrete benefits:

1. **Sample efficiency**: an equivariant model with `P` parameters effectively gets to see `P × |G|` examples (where `|G|` is the group order — infinite for SO(2), so this is morally a "huge" effective dataset). You're not learning to handle each rotation separately.

2. **Out-of-distribution generalization**: rotations that don't appear at training time still work. A non-equivariant model would have to be data-augmented to handle them, and even then would be merely *approximately* equivariant.

3. **Compositional structure**: the latent is structurally meaningful. You can ask "is this latent rotation-invariant at this channel?" and get a mathematical answer, not just an empirical one.

### Where equivariance might hurt your model

If your data has **rotational asymmetry** (e.g., always-aligned tissue sections, gravity-dependent biological structures, scanner orientation), equivariance is hostile to learning it. Your model literally cannot learn "T-cells always point upward" because that's a violation of rotational symmetry.

For IMC tissue data, this is probably *not* a concern — slides are mounted at arbitrary orientations and biological structures don't have a global preferred direction at the cell scale. But it's worth verifying that you're not losing anything by enforcing equivariance.

### How does the EquivariantHyperkernel specifically help?

It solves a specific problem: **how to combine information across markers without breaking equivariance.**

Without an equivariant marker mixer, you'd have to either:
- (a) Use a non-equivariant mixer → equivariance is lost at the MA→PM interface (this was v1)
- (b) Use a separate equivariant convolution per marker → loses the marker-conditional flexibility
- (c) Use shared parameters across markers → loses marker-specific behavior

The hyperkernel achieves **all three properties simultaneously**: equivariant, marker-conditional, and shared backbone. The trick is that scalar weights commute with the rotation action, so marker conditioning (which is rotation-independent — markers don't rotate) and rotation equivariance (which acts on the spatial+field structure) are *orthogonal axes of variation* that don't interfere.

This is genuinely elegant. It's the kind of architectural choice that wouldn't be obvious without thinking carefully about the group-theoretic structure.

**Concrete benefit**: every additional marker you add costs `K_in × K_out` new parameters in the embedding (cheap) and doesn't disturb the equivariance of the existing markers. You can train on any subset of markers and the equivariance properties are preserved.

---

## 3. The Trivial Projection: A Scalar Codebook?

**Yes, your intuition is correct.** Projecting to trivial scalars is essentially building a **rotation-invariant codebook of features**, applied per-spatial-location. Here's why this is *more than ok* — it's actually principled.

### The codebook interpretation

Your final latent is `[B, output_scalars, 15, 15]` where each spatial position carries `output_scalars` rotation-invariant scalars. Think of these scalars as a **dictionary of rotation-invariant cellular descriptors**:

- Channel 1 might encode "presence of T-cell-like morphology"
- Channel 2 might encode "stromal density"
- Channel 3 might encode "edge density / structural complexity"
- Channel k might encode "co-localization of CD3 and CD8"

Each of these descriptors is intrinsically rotation-invariant — it doesn't matter how the image was rotated, the answer should be the same. So forcing the channel dimension to be invariant is *semantically appropriate* for these kinds of descriptors.

### Why the spatial equivariance matters

The latent is `15×15` spatial. So the **spatial layout** of the codebook activations is rotation-equivariant: if you rotate the input, you get the same codebook entries activating, just at rotated spatial positions.

This decouples two things cleanly:
- **What** is at each location: rotation-invariant scalar (channel)
- **Where** each location is: rotation-equivariant spatial position

This decomposition is **exactly what you want for downstream tasks**:

- **Cell type classification**: needs the "what" (the channel descriptors), should ignore "where". Easy from your latent.
- **Spatial pattern analysis**: needs the "where", should be insensitive to rotation. Also easy from your latent.
- **Reconstruction**: needs both. The decoder reads the spatial pattern of channel descriptors and reconstructs accordingly.

### The information-theoretic argument

You might worry: "by projecting to trivial, am I throwing away the freq-1, freq-2, ... components I worked so hard to build?"

Two responses:

1. **The higher-frequency components are processed inside the encoder**, not just at the end. Every convolution in the BL-regular pipeline mixes frequency components in a controlled way; the freq-2 information is not "wasted" — it's used to construct the final trivial scalars through Clebsch-Gordan combinations. The Regular2Trivial layer is a learned 1×1 R2Conv that *picks the optimal trivial-rep combination* from the rich field representation.

2. **Spatial structure preserves orientation information**. Even though each pixel is a scalar, the spatial pattern of those scalars encodes orientation. A horizontally-elongated cell produces a horizontally-elongated pattern of channel activations; rotated 90° it produces a vertically-elongated pattern. The decoder can read this orientation from the spatial layout without needing intra-pixel orientation channels.

### When this would NOT be ok

The codebook projection *would* be problematic if:
- You needed **sub-pixel orientation information** (e.g., the orientation of a structure smaller than your downsampling factor, ~7.5px in your case)
- You needed **multiple orientations to coexist at one pixel** (e.g., a junction where two vessels cross at different angles)
- Your downstream task explicitly required **per-pixel orientation labels** (e.g., predicting cell polarity vectors)

For virtual staining and most cellular characterization tasks, none of these apply at the latent resolution. Your `15×15` latent has each pixel covering ~`7.5×7.5` original pixels — about the size of a single cell. Per-pixel orientation isn't needed at that resolution.

### The codebook size question

Your `output_scalars` choice (e.g., 512) sets the codebook size. This is a hyperparameter worth thinking about:

- Too small (e.g., 32) → not enough invariant descriptors to represent all cell types and contexts
- Too large (e.g., 2048) → wasteful, redundant channels, harder to interpret
- "Right" size → roughly the number of biologically distinct local cellular contexts, which for IMC is probably a few hundred (cell type × activation state × micro-environment combinations)

512 seems reasonable. You could test by training with `output_scalars ∈ {128, 256, 512, 1024}` and measuring downstream task performance — there should be a saturation point where adding more channels doesn't help.

### The "encoding equivariant features into scalars" framing

Your phrasing — "encoding equivariant features into a scalar codebook" — captures something important: the **equivariant computation has done work for you**. The trivial scalars at the end are not the same as scalars you'd get from a non-equivariant CNN. They are scalars that:

- Were computed from features that respected rotational symmetry throughout
- Represent semantically rotation-invariant descriptors by *construction*, not by accident
- Have the property that rotating the input produces the same scalars at corresponding rotated locations

A non-equivariant CNN's "scalars" don't have this property. Two patches that are rotations of each other can produce arbitrarily different scalar features. Yours can't (up to numerical error from discretization).

So yes, "scalar codebook learned through an equivariant pipeline" is a correct and useful framing. The equivariant intermediate computation is what makes the final scalars *meaningful and consistent*, not what makes them "more expressive" in a parameter-count sense.

---

## 4. Synthesis: Is Your Architecture Sensible?

Let's evaluate the design choices against the analysis above:

| Design choice | Justification |
|---|---|
| Equivariant MA encoder | Marker-agnostic features should be rotation-invariant in their *meaning* — equivariance enforces this structurally. ✓ |
| EquivariantHyperkernel for marker mixing | Cleanest way to combine marker conditioning with rotation equivariance. Scalar-per-pair is a defensible starting point. ✓ |
| Equivariant PM encoder | Spatial pattern aggregation should respect rotational symmetry of tissue context. ✓ |
| Antialiased downsampling | Reduces aliasing under continuous rotations — directly addresses the main source of equivariance error. ✓ |
| Final trivial projection | Builds a rotation-invariant codebook with rotation-equivariant spatial layout — clean decomposition for downstream tasks. ✓ |
| Non-equivariant decoder | Decoder reads orientation from spatial layout; doesn't need to be equivariant. Saves compute and complexity. ✓ |

### The one place I'd push back

The **scalar-per-field-pair hyperkernel** is the least expressive equivariant parameterization. If you ever find that:
- The model underfits relative to a non-equivariant baseline
- Reconstruction quality plateaus and won't improve with more parameters elsewhere
- Equivariance error is good but reconstruction is mediocre

...then upgrade to a full intertwiner basis (`R×` more parameters, full equivariant expressivity). But based on your reported results — comparable virtual staining + good equivariance — you don't seem to need this. The simple version is doing its job.

### The architecture in one sentence

You've built a model that **factors cellular morphology into rotation-invariant content (channels) and rotation-equivariant context (spatial layout)**, processed through an equivariant pipeline, and reconstructed by a non-equivariant decoder that reads orientation from spatial structure.

That's a clean and principled design. The fact that it works as well as the non-equivariant baseline on virtual staining while having structural equivariance properties is the bar for "this was worth building." You cleared it.

---

## 5. Reading Recommendations

If you want to deepen your understanding of why this design works:

1. **Cohen & Welling, "Group Equivariant Convolutional Networks" (2016)** — the foundational paper on equivariant CNNs; clearest explanation of the inductive bias argument.

2. **Weiler & Cesa, "General E(2)-Equivariant Steerable CNNs" (2019)** — the paper behind ESCNN's design, including the BL-regular representation and its theoretical basis.

3. **Worrall et al., "Harmonic Networks: Deep Translation and Rotation Equivariance" (2017)** — shows the relationship between continuous rotation equivariance and circular harmonics; directly relevant to your max_freq choice.

4. **Bekkers et al., "Roto-Translation Covariant Convolutional Networks for Medical Image Analysis" (2018)** — application of equivariant nets to histology, addresses exactly the trade-off you've made.

The key takeaway from all of these: equivariance is a way of *encoding domain knowledge* into the architecture. You've done that correctly for the assumption that "tissue at a microscopic scale has no preferred orientation," which is a defensible biological assumption for IMC data.
