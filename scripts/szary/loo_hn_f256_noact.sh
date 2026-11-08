#!/bin/bash
#SBATCH --job-name=loo-f256-noact
#SBATCH --partition=common
#SBATCH --qos=szlukasik
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=logs/loo_hn_f256_noact_%j.out
#SBATCH --error=logs/loo_hn_f256_noact_%j.err
# Leave-one-out inpainting on the hn test split for the V3-256 transition-nonlinearity
# ablation, with exactly the protocol of loo_hn_f256.sh (deterministic all-channel LOO,
# --patch-mask-ratio 0, scored at 128 px; architecture read from the checkpoint).
#   EPOCH=29 sbatch scripts/szary/loo_hn_f256_noact.sh       (repo root, VENV exported)
# Afterwards: python scripts/szary/loo_aggregate.py 29 f256_noact
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${VENV:?set VENV to the venv directory}/bin/activate"
EPOCH=${EPOCH:-29}
CKPT=checkpoints_equiv_zs_hn_f256_noact/checkpoint-equiv-v3-int64-zs-hn-f256-noact-epoch_${EPOCH}.pth
OUT=validation_results/loo_hn_f256_noact_e${EPOCH}
MAPS=/storage_ssd_1/szlukasik/loo_maps/loo_hn_f256_noact_e${EPOCH}.npz
[ -f "$CKPT" ] || { echo "missing $CKPT"; exit 1; }
ls -la "$CKPT"
export PYTORCH_ALLOC_CONF=expandable_segments:True
python run_validation.py \
    --config configs/train_equivariant_v3_zs_hn_f256_noact_szary_config.yaml \
    --checkpoint "$CKPT" \
    --panel-config configs/all_panels_config_szary_hn_only.yaml \
    --all-channels-loo \
    --output-dir "$OUT" \
    --save-maps-path "$MAPS"
echo "done $(date)"
