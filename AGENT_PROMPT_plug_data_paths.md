# Coding-agent task: wire in the nsclc2 raw labelled data and run the equivariant cell-typing eval

## Context

This repo (`celltyping-downstream/`, branch `virtual-staining` of
`dav3794/multiplex-image-model`) runs the **cell-typing downstream task**.
The integration for the **equivariant ConvNeXt autoencoder** is already
written and committed:

- `celltyping/core/models/immuvis_equivariant.py` — model setup hook
  (`setup_immuvis_equivariant`): builds `EquivariantMultiplexAutoencoder`
  from the training YAML, strict-loads the checkpoint, self-contained
  arcsinh→butterworth→clip preprocessing (no OpenCV/VirTues).
- `celltyping/core/embeddings/inference_equivariant.py` — VirTues-free
  embedding runner (`ProcessedCellDataset` reads the processed dir; no ESM-2,
  no `../virtues`).
- `celltyping/core/models/registry.yaml` — entry `equiv_convnext_v2`.
- `celltyping/gold_standard/gs_config.yaml` — gold-standard config with
  `<<<TODO>>>` `raw:` placeholders.
- `run_equivariant_celltyping.sh` — the 3-step pipeline.

The model checkpoint + training config + tokenizer/panel configs live in the
sibling training repo and are referenced by absolute path in
`registry.yaml` (`equiv_convnext_v2`). **On a different cluster those absolute
paths will differ — update them too.**

## The blocker you must resolve

The gold-standard (`nsclc2-panel1`) **raw labelled data is not present**.
The pipeline needs four inputs (consumed by `celltyping/gold_standard/data.py`
+ `celltyping/core/data.py`):

1. `img_dir` — raw IMC multi-channel tiffs, one per tissue, named
   `{tissue_id}.tiff`, shape `(C, Y, X)`.
2. `mask_dir` — instance segmentation masks, one per tissue,
   `{tissue_id}.tiff`, integer cell IDs, shape `(Y, X)`.
3. `rds_path` — labelled single-cell experiment `.rds`. Read via
   `rds2py.read_rds` → `as_summarized_experiment` → `.to_anndata()[0].obs`,
   which **must** contain columns: `image`, `Pos_X`, `Pos_Y`, `celltypes`
   (tissue_id is derived as `image` with `.tiff` stripped).
4. `channels_csv_path` — markers CSV with columns `Marker Name`,
   `protein_id`; row order must match the tiff channel order, and length
   must equal the tiff channel count.

## Your tasks

1. **Locate the nsclc2-panel1 raw labelled data on this cluster.** Search
   broadly (group/shared dirs, archives, other accounts you can read).
   The IMMUCan-derived source originally lived under
   `/raid_encrypted/immucan/IMC/immu-vis-other-ds/...` on the origin
   machine — look for a transferred copy or equivalent. Verify each
   candidate actually satisfies the four-input contract above (open a tiff,
   open a mask, load the `.rds` and check the obs columns).

2. **Edit `celltyping/gold_standard/gs_config.yaml`:** replace the four
   `<<<TODO>>>` `raw:` paths with the real ones, and set `base_path` /
   `processed_dir` to a writable location on this cluster (plenty of space;
   the processed dir holds per-cell crops).

3. **Fix the absolute paths in `celltyping/core/models/registry.yaml`**
   under `equiv_convnext_v2` (`repo_path`, `checkpoint_path`, `conf`,
   `panel_conf_path`, `tokenizer_path`) to point at this cluster's copy of
   the training repo + the `J2598224 epoch_79` checkpoint and the
   `train_masked_equivariant_config_flip_v2_wider_modelv2.yaml` config.
   The checkpoint **must** match that config (strict `load_state_dict`).

4. **Pick the Python environment.** Do NOT `uv sync` this project (its
   `rds2py==0.4.0` builds from C++ source and needs `g++`/`module load
   GCC`). Prefer an existing env that already has `torch` + `escnn` +
   `scikit-learn`. `rds2py`+`anndata` are needed ONLY for step 1 of the
   pipeline (reading the `.rds`); if the chosen env lacks them, build a
   tiny separate env just for `python -m gold_standard.data`, or `module
   load` a GCC and `pip install rds2py anndata` into a scratch venv.

5. **Run the pipeline** (see `run_equivariant_celltyping.sh`; fix the
   `VENV`/`PROJ` paths in it for this cluster):
   ```
   python -m gold_standard.data
   python -m core.embeddings.inference_equivariant \
     --config ./gold_standard/gs_config.yaml --model equiv_convnext_v2 \
     --scheme patch --batch_size 32 --split test
   python -m core.crossval \
     --config ./gold_standard/gs_config.yaml --run equiv_convnext_v2_patch
   ```
   Run `python -m core.embeddings.inference_equivariant` from the
   `celltyping/` project dir with `PYTHONPATH` including it. Use a GPU node
   (the escnn model is slow on CPU).

## Verify / watch for

- **Strict load**: `load_state_dict(strict=True)` fails if the checkpoint
  doesn't match the training YAML — confirm config↔checkpoint pairing first.
- **Tokenizer alignment**: `nsclc2-panel1` markers (minus `DNA1`/`DNA2`) are
  mapped through `configs/all_markers_tokenizer.yaml` from the training repo;
  every marker in `all_panels_config.yaml`'s `markers: nsclc2-panel1` list
  must exist in that tokenizer (it must be the *same* tokenizer the model was
  trained with). The markers CSV channel order must match the tiff.
- **Crop size**: cell crops are 32×32 (`core/data.py` `patch_size`); the
  equivariant encoder downsamples ~8× with antialiased pooling. If the
  encoder errors or feature maps collapse at 32×32, set `input_size: 64`
  (or `113`) under `equiv_convnext_v2` in `registry.yaml` (the hook
  bilinearly upsamples crops before encoding).
- **Sanity first**: run `_sanity_check.py` (no real data needed) — it
  builds the hook, strict-loads the checkpoint, and runs `encode_images`
  on synthetic 32×32 and 64×64 crops. It must print `SANITY DONE` with
  `[ok] encode_images` lines before you trust a full run.
- The crossval `--run` value must equal the embeddings subdir name
  (`{model}_{scheme}` = `equiv_convnext_v2_patch`).
- Cell-type labels come from the `.rds`; `celltypes_map.json` is written by
  `gold_standard.data` into `processed_dir` and consumed by `core.crossval`.

## Definition of done

`core.crossval` prints per-class accuracy / balanced-accuracy / F1 / AUC for
the equivariant model on nsclc2-panel1, and writes results under
`$base_path/crossval/`.
