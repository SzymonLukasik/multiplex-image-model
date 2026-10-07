#!/bin/bash
#SBATCH --job-name=prof-f256
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/prof_f256_%j.out
#SBATCH --error=logs/prof_f256_%j.err
# Add --account / --partition lines here if szary's SLURM needs them.
#
# Memory + speed profile of the 256-field (ma=1) equivariant V3 on one GPU, with
# synthetic input (no dataset needed). Run this BEFORE the training chain.
#   * memory: full step (fwd + bwd + AdamW) at batch 4, 2 and 1, for 28/40/47 markers;
#     47 is the largest panel, so "PEAK +opt step" at C=47 is what must fit.
#   * speed : seconds per micro-step at batch 4, projected hours per epoch.
# Submit from the repo root:  mkdir -p logs && VENV=/path/to/venv sbatch scripts/szary/profile_f256.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${VENV:?set VENV to the venv directory}/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True
CFG=configs/train_equivariant_v3_zs_hn_f256_szary_config.yaml

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch, escnn; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| escnn', escnn.__version__)"

for B in 4 2 1; do
    python _profile_gpu_memory.py "$CFG" --channels 28 40 47 --batch "$B"
done
python _time_train_step.py "$CFG" --batch 4 --channels 40 --steps 30 --iters-per-epoch 9128
