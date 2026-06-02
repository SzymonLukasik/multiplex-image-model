#!/bin/bash
#SBATCH --job-name=smoke-trainval-rf
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgspacelet2-gpu-a100
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00-01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=logs/smoke_trainval_rf_%j.out
#SBATCH --error=logs/smoke_trainval_rf_%j.err

source $SCRATCH/venvs/immu-vis/bin/activate

# Smoke test: training val loop + discrete rotation/flip equivariance probe.
#
# Same dataloader + masking + plot as smoke_test_trainval, plus:
#   * one figure per (sample, T) with col-3 replaced by T⁻¹(D(E(T(x_masked))))
#     and per-channel consistency MSE shown in the col-3 title,
#   * one aggregate figure per run showing per-sample bars of across-channel
#     consistency for each transform.
#
# Defaults target the model from logs/mask_v2_ddp_2506143.out (config's native
# model_type EquivariantConvnext -> v1 module that matches the pre-redesign
# checkpoint).
#
# Env overrides:
#   CFG=...           config YAML        (default: flip_v2_wider)
#   CKPT=...          checkpoint         (default: J2506143 epoch_199)
#   MODEL_TYPE=...    model_type override (default: none -> config's value)
#   OUT=...           output dir (default auto-suffixed by INPUT_SIZE if set)
#   NUM_PLOTS=N       number of val batches to plot
#   TRANSFORMS=...    space-separated list (default: rot90 rot180 rot270 hflip)
#   INPUT_SIZE=N      override config.input_image_size — use 128 to bypass the
#                     [3:-4] decoder-output crop (16·8=128 → no crop) and
#                     compare against the training-time 113. Leave unset to
#                     use the config's value.
#   NUM_LATENT_CH=N   latent channels per latent figure (default 6)

CFG=${CFG:-train_masked_equivariant_config_flip_v2_wider.yaml}
CKPT=${CKPT:-checkpoints/checkpoint-EquivariantConvnext_v2_20260403_101452_J2506143-epoch_199.pth}
MODEL_TYPE=${MODEL_TYPE:-}
INPUT_SIZE=${INPUT_SIZE:-}
NUM_LATENT_CH=${NUM_LATENT_CH:-6}
NUM_PLOTS=${NUM_PLOTS:-5}
TRANSFORMS=${TRANSFORMS:-rot90 rot180 rot270 hflip}

if [ -n "$INPUT_SIZE" ]; then
    OUT=${OUT:-smoke_test_trainval_rf_size${INPUT_SIZE}}
else
    OUT=${OUT:-smoke_test_trainval_rf}
fi

MT_ARG=""
if [ -n "$MODEL_TYPE" ]; then MT_ARG="--model-type $MODEL_TYPE"; fi
IS_ARG=""
if [ -n "$INPUT_SIZE" ]; then IS_ARG="--input-size $INPUT_SIZE"; fi

srun python ./smoke_test_trainval_rf.py \
    --config "$CFG" \
    --checkpoint "$CKPT" \
    $MT_ARG \
    $IS_ARG \
    --output-dir "$OUT" \
    --num-plots "$NUM_PLOTS" \
    --num-latent-channels "$NUM_LATENT_CH" \
    --transforms $TRANSFORMS
