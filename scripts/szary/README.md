# V3-256 on szary (half-width equivariant V3, 1 GPU)

Width-bracket run for the equivariant V3. It differs from
`configs/train_equivariant_v3_ddp_zs_hn_config.yaml` **only in encoder width**:

| | V3 (athena) | V3-256 (this run) |
|---|---|---|
| `pm_embedding_dims` | [128, 256, 512] | [64, 128, 256] |
| hyperkernel `embedding_dim` | 128 | 64 |
| GPUs × batch × accumulation | 2 × 4 × 1 = 8 | 1 × 4 × 2 = 8 |
| everything else | — | identical (`ma_layers_blocks [1]`, decoder `[2, 2, 1]`, β = 0.5, lr schedule, masking, data, internal 64) |

Its pooled invariant readout is 6 × 256 = **1536 dims**, the same as van1536, so the
downstream comparisons need no PCA width matching.

Config: `configs/train_equivariant_v3_zs_hn_f256_szary_config.yaml`

## 1. Clone

```bash
git clone git@github.com:SzymonLukasik/multiplex-image-model.git mim-f256   # or the https URL
cd mim-f256
git checkout lukass/equiv-f256-szary
```

## 2. Environment (Python 3.12)

szary's driver is 565.57 (CUDA 12.7), so install the **cu126** PyTorch build. The cu128
wheels used on athena expect a newer driver.

```bash
uv venv --python 3.12 ~/venvs/mim-f256 && source ~/venvs/mim-f256/bin/activate
uv pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install "escnn @ git+https://github.com/QUVA-Lab/escnn.git@9ad44cc37d694d9c805c4fef4d16722a254b3bf2"
uv pip install -r scripts/szary/requirements-szary.txt   # exact athena versions
uv pip install -e . --no-deps   # pyproject pins differ from the validated venv; --no-deps keeps the pins above
python -c "import torch, escnn; print(torch.__version__, torch.cuda.is_available(), escnn.__version__)"
```

If `py3nj` or `lie_learn` fail to build, install `gfortran`, or install lie_learn from
`git+https://github.com/AMLab-Amsterdam/lie_learn`.

## 3. Data (~886 GB)

Training needs the exact same arcsinh-transformed split as athena
(`immuvis_split_patches_onlyarcsinh`, train + test). Check the free space first
(`df -h`), then copy it from athena:

```bash
rsync -a --info=progress2 \
  plgslukasik@athena.cyfronet.pl:/net/tscratch/people/plgslukasik/immuvis_split_patches_onlyarcsinh/ \
  /DATA/ON/SZARY/immuvis_split_patches_onlyarcsinh/
```

Then put that directory into the panel config (both `train` and `test`):

```bash
sed -i 's|/CHANGE_ME|/DATA/ON/SZARY|' configs/all_panels_config_szary_zs_hn.yaml
```

The training script refuses to start while `CHANGE_ME` is still there or a path is missing.

## 4. Profile first (synthetic input, no data needed, ~15 min)

```bash
mkdir -p logs
VENV=~/venvs/mim-f256 sbatch scripts/szary/profile_f256.sh
```

Read `logs/prof_f256_<job>.out`:
* **`PEAK +opt step` at C = 47** (the largest panel) must stay below about 30 GiB at
  batch 4, the configured setting (4 × 2 accumulation). If it doesn't, set `batch_size: 2`
  and `gradient_accumulation_steps: 4` (same effective batch), or 1 × 8 if even that is
  too tight.
* **Projected h/epoch** is a lower bound, since data loading isn't included.

Add `#SBATCH --account/--partition` lines to both scripts if szary needs them.

## 5. Train as a dependency chain

Each link resumes from the last checkpoint (`from_checkpoint: last`).

```bash
jid=$(VENV=~/venvs/mim-f256 sbatch --parsable scripts/szary/train_equiv_f256.sh)
for i in $(seq 2 8); do
  jid=$(VENV=~/venvs/mim-f256 sbatch --parsable --dependency=afterany:$jid scripts/szary/train_equiv_f256.sh)
done
```

QOS on szary is 2 running jobs / 2 GPUs / 16 CPUs. This chain holds one GPU and
8 CPUs, so one slot stays free for evaluation jobs.

Numbered checkpoints are kept every 10 epochs (`epoch_9, 19, 29, …`), matching the
epochs V3 was evaluated at (29, 49, 79 for reconstruction, 79 for cell typing, 159 for
clinical). Evaluate them with the same zero-shot hn LOO protocol as V3.
