#!/bin/bash
#SBATCH --job-name=test-noact
#SBATCH --partition=common
#SBATCH --qos=szlukasik
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=logs/test_noact256_%j.out
#SBATCH --error=logs/test_noact256_%j.err
# GPU pre-launch checks for the V3-256 transition-nonlinearity ablation
# (tests_noact/test_noact_v3_256.py, sections a-d). Submit from the repo root, VENV exported.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${VENV:?set VENV to the venv directory}/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True
ONLY=${ONLY:-a,b,c,d} python tests_noact/test_noact_v3_256.py
echo "done $(date)"
