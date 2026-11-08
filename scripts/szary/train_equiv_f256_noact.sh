#!/bin/bash
#SBATCH --job-name=eq-f256-noact
#SBATCH --partition=common
#SBATCH --qos=szlukasik
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/train_equiv_f256_noact_%j.out
#SBATCH --error=logs/train_equiv_f256_noact_%j.err
#
# TRANSITION-NONLINEARITY ABLATION of V3-256 (zero-shot hn): pool_use_act=false and
# use_hyperkernel_act=false, everything else identical to V3-256. Pre-registered
# reading: configs/README_f256_noact.md.
#
# Same chaining as train_equiv_f256.sh: the config sets `from_checkpoint: last` and a
# last-checkpoint is written every epoch, so each link resumes where the previous one
# hit the wall clock (10 days is the QOS maximum). Numbered checkpoints every 10 epochs.
#
# GPU count is read from the allocation, so the same script serves both configs:
#   sbatch scripts/szary/train_equiv_f256_noact.sh                 # 1 GPU, accumulation 2
#   sbatch --gpus=2 --cpus-per-task=16 scripts/szary/train_equiv_f256_noact.sh   # 2-GPU DDP
# Submit from the repo root with VENV exported (see scripts/szary/README.md).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${VENV:?set VENV to the venv directory}/bin/activate"

NGPU=${SLURM_GPUS_ON_NODE:-1}
if [ "$NGPU" -ge 2 ]; then
    CFG=configs/train_equivariant_v3_zs_hn_f256_noact_szary_2gpu_config.yaml
else
    CFG=configs/train_equivariant_v3_zs_hn_f256_noact_szary_config.yaml
fi
echo "GPUs: $NGPU  config: $CFG"

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
enc = cfg["encoder"]
assert enc["pool_use_act"] is False and enc["use_hyperkernel_act"] is False, "not the noact config"
print("data paths OK:", paths, "| pool_use_act", enc["pool_use_act"], "use_hyperkernel_act", enc["use_hyperkernel_act"])
PY

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export PYTORCH_ALLOC_CONF=expandable_segments:True
export COMET_OFFLINE_DIRECTORY="${PWD}/.cometml-offline"
mkdir -p "$COMET_OFFLINE_DIRECTORY" "$(readlink -m checkpoints_equiv_zs_hn_f256_noact)"   # symlink -> /storage_ssd_1 (szary only)

srun torchrun \
    --nnodes=1 \
    --nproc_per_node="$NGPU" \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_masked_model_ddp.py "$CFG"
