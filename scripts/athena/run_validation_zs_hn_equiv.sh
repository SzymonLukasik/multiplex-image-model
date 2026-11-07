#!/bin/bash
#SBATCH --account=plgspacelet2-gpu-a100
#SBATCH --job-name=val-equiv-hn
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=00-06:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --output=logs/val_equiv_zs_hn_%j.out
#SBATCH --error=logs/val_equiv_zs_hn_%j.err

# ZERO-SHOT leave-one-out evaluation of the FULLY-EQUIVARIANT V3 model
# (jobs 3034057 / 3109482, run equiv-v3-int64-zs-hn) on the held-out 'hn' dataset,
# at the SAME epoch (29) as scripts/run_validation_zs_hn_van_progdec.sh.
#
# See that script for the protocol rationale; everything here is identical except the
# checkpoint and the resources. Pure leave-one-out: one channel fully removed and
# predicted, every channel of every image, remaining channels left fully dense.
#
# Resources are larger than the vanilla partner for two reasons: the checkpoint is
# 10.8 GB (escnn stores expanded filter buffers alongside the real parameters), and
# escnn layers build large intermediates that scale with batch size — hence the
# small forward chunk. Drop FWD_BATCH to 2 or 1 if it OOMs.

source /net/tscratch/people/plgslukasik/venvs/immu-vis-fresh/bin/activate
cd /net/tscratch/people/plgslukasik/immu-vis/multiplex-image-model/multiplex-image-model-fresh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE=/net/tscratch/people/plgslukasik/immu-vis/multiplex-image-model
CFG=${CFG:-configs/train_equivariant_v3_ddp_zs_hn_config.yaml}
CKPT=${CKPT:-$BASE/checkpoints_equiv_zs_hn/checkpoint-equiv-v3-int64-zs-hn-epoch_29.pth}
PANEL=${PANEL:-configs/all_panels_config_tscratch_hn_only.yaml}
TAG=${TAG:-equiv_e29}
FWD_BATCH=${FWD_BATCH:-4}
# Set LIMIT="--limit-images 2" for a fast pipeline smoke test.
LIMIT=${LIMIT:-}
# EXTRA forwards extra flags, e.g. EXTRA="--internal-resolution" to score at
# the model's internal 64 instead of the 128 input resolution.
EXTRA=${EXTRA:-}
# Spatial masking of the KEPT channels. 0.0 = clean context (the intended protocol).
RATIO=${RATIO:-0.0}

# WAIT_FOR_CKPT: seconds to wait for $CKPT to appear and finish being written.
# This exists so an eval can be queued for an epoch the training job has NOT
# reached yet. Queue waits here run to days, so the scarce resource is the
# scheduling slot, not GPU minutes - failing instantly because we arrived 20
# minutes early would throw that slot away. Pair it with --begin=<eta>.
#
# The size-stability check matters: this checkpoint is 10.8 GB, so it exists on
# disk for minutes while still incomplete, and torch.load on a partial file dies
# with an opaque zip error that looks like a corrupt model.
if [ -n "${WAIT_FOR_CKPT:-}" ]; then
    echo "Waiting up to ${WAIT_FOR_CKPT}s for $CKPT"
    deadline=$((SECONDS + WAIT_FOR_CKPT))
    while [ ! -f "$CKPT" ]; do
        if [ $SECONDS -gt $deadline ]; then
            echo "ERROR: $CKPT never appeared within ${WAIT_FOR_CKPT}s" >&2
            exit 1
        fi
        sleep 120
    done
    prev=-1
    while :; do
        cur=$(stat -c %s "$CKPT")
        if [ "$cur" = "$prev" ] && [ "$cur" -gt 0 ]; then break; fi
        if [ $SECONDS -gt $deadline ]; then
            echo "ERROR: $CKPT still growing after ${WAIT_FOR_CKPT}s" >&2
            exit 1
        fi
        prev=$cur
        sleep 60
    done
    echo "Checkpoint ready ($(stat -c %s "$CKPT") bytes) after ${SECONDS}s"
fi

OUTDIR=$BASE/validation_results_zs_hn_${TAG}_r${RATIO}
echo "=============== $TAG  patch-mask-ratio $RATIO ==============="
srun python -u run_validation.py \
    --config "$CFG" \
    --checkpoint "$CKPT" \
    --panel-config "$PANEL" \
    --model-type fully_equivariant_v3 \
    --all-channels-loo \
    --forward-batch-size $FWD_BATCH \
    --patch-mask-ratio $RATIO \
    --mask-patch-size 8 \
    --save-maps-path "$OUTDIR/pred_maps.npz" \
    --output-dir "$OUTDIR" $LIMIT $EXTRA || exit 1
echo "DONE"
