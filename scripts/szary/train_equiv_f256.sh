#!/bin/bash
#SBATCH --job-name=eq-f256
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_equiv_f256_%j.out
#SBATCH --error=logs/train_equiv_f256_%j.err
# Add --account / --partition lines here if szary's SLURM needs them, and raise
# --time to the partition maximum if it allows more than 24 h.
#
# 1-GPU training of the 256-field (ma=1) fully-equivariant V3 (zero-shot hn).
# Meant to be submitted as a dependency chain: the config sets
# `from_checkpoint: last`, and a last-checkpoint is written every epoch, so each
# link resumes where the previous one hit the wall clock. Numbered checkpoints are
# kept every 10 epochs (epoch_9, 19, 29, ... - the epochs V3 was evaluated at).
# Submit from the repo root (see scripts/szary/README.md).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${VENV:?set VENV to the venv directory}/bin/activate"
CFG=configs/train_equivariant_v3_zs_hn_f256_szary_config.yaml

# Fail fast if the panel config still has the placeholder or the data is missing.
python - "$CFG" <<'PY'
import os, sys
from ruamel.yaml import YAML
y = YAML(typ="safe")
cfg = y.load(open(sys.argv[1]))
paths = y.load(open(cfg["panel_config"]))["paths"]
bad = {k: v for k, v in paths.items() if "CHANGE_ME" in str(v) or not os.path.isdir(str(v).strip())}
if bad:
    sys.exit(f"panel config {cfg['panel_config']} has missing data paths: {bad}")
print("data paths OK:", paths)
PY

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export PYTORCH_ALLOC_CONF=expandable_segments:True
export COMET_OFFLINE_DIRECTORY="${PWD}/.cometml-offline"
mkdir -p "$COMET_OFFLINE_DIRECTORY" checkpoints_equiv_zs_hn_f256

srun torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_masked_model_ddp.py "$CFG"
