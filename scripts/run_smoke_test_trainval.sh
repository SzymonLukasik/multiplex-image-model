#!/bin/bash
#SBATCH --job-name=smoke-trainval
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgspacelet2-gpu-a100
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00-03:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=logs/smoke_trainval_%j.out
#SBATCH --error=logs/smoke_trainval_%j.err

source $SCRATCH/venvs/immu-vis/bin/activate

# Smoke test that mirrors the training val loop: same dataloader, same
# random-channel-drop + spatial-patch masking (using the config's masking
# knobs), same [3:-4] decoder-output crop, same plot_reconstructs_with_uncertainty
# triplet figure (Original / Reconstructed / Variance) that gets logged to
# TensorBoard during training — but written to PNG files here.
#
# Defaults target the model from logs/mask_v2_ddp_2506143.out (config's native
# model_type EquivariantConvnext -> multiplex_model.equivariant_modules v1
# module, which matches that pre-redesign checkpoint).
#
# Env overrides:
#   CFG=...           config YAML        (default: flip_v2_wider)
#   CKPT=...          checkpoint         (default: J2506143 epoch_199)
#   MODEL_TYPE=...    model_type override (default: none -> config's value)
#   OUT=...           output dir
#   NUM_PLOTS=N       number of val batches to plot

CFG=${CFG:-train_masked_equivariant_config_flip_v2_wider.yaml}
CKPT=${CKPT:-checkpoints/checkpoint-EquivariantConvnext_v2_20260403_101452_J2506143-epoch_199.pth}
MODEL_TYPE=${MODEL_TYPE:-}
OUT=${OUT:-smoke_test_trainval}
NUM_PLOTS=${NUM_PLOTS:-5}

MT_ARG=""
if [ -n "$MODEL_TYPE" ]; then MT_ARG="--model-type $MODEL_TYPE"; fi

srun python ./smoke_test_trainval.py \
    --config "$CFG" \
    --checkpoint "$CKPT" \
    $MT_ARG \
    --output-dir "$OUT" \
    --num-plots "$NUM_PLOTS"
