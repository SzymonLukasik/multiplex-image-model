#!/bin/bash
#SBATCH --account=plgspacelet2-gpu-a100
#SBATCH --job-name=eq-f256-fourier
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=02-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --signal=B:USR1@900
#SBATCH --output=logs/train_equiv_ddp_zshn_f256_fourier_%j.out
#SBATCH --error=logs/train_equiv_ddp_zshn_f256_fourier_%j.err

# FOURIER-ELU ABLATION of V3-256 (EquiVis-256): every NormNonLinearity in the
# ENCODER replaced by escnn FourierELU (N=8 rotations x 2 flips). Everything else
# identical to V3-256 (half-width diff of train_equivariant_v3_ddp_zs_hn_config.yaml).
# Pre-registered reading: checkpoints_equiv_zs_hn_f256_fourier/README.md.
# 2 GPUs x 4 = effective batch 8. Submitted as a dependency chain; from_checkpoint:
# last resumes each link.
source /net/tscratch/people/plgslukasik/venvs/immu-vis-fresh/bin/activate
cd /net/tscratch/people/plgslukasik/immu-vis/multiplex-image-model/multiplex-image-model-fresh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export PYTORCH_ALLOC_CONF=expandable_segments:True
export COMET_OFFLINE_DIRECTORY=/net/tscratch/people/plgslukasik/immu-vis/multiplex-image-model/.cometml-offline
mkdir -p "$COMET_OFFLINE_DIRECTORY" \
         /net/tscratch/people/plgslukasik/immu-vis/multiplex-image-model/checkpoints_equiv_zs_hn_f256_fourier

# Self-chaining (scripts/selfchain_lib.sh): the next link is submitted only in the
# last 15 min of this one, so the run holds a single job in the queue.
CHAIN_SCRIPT=/net/tscratch/people/plgslukasik/immu-vis/multiplex-image-model/scripts/train_equiv_v3_ddp_zs_hn_f256_fourier.sh
CHAIN_DONE_FILE=/net/tscratch/people/plgslukasik/immu-vis/multiplex-image-model/checkpoints_equiv_zs_hn_f256_fourier/final_model-equiv-v3-f256-fourier-zs-hn.pth
CHAIN_MAX_LINKS=8
source /net/tscratch/people/plgslukasik/immu-vis/multiplex-image-model/scripts/selfchain_lib.sh

# Zero-shot LOO of any new epoch checkpoints runs inside this allocation (no
# separate eval job): before training, and once more if training finishes.
EVAL_PENDING="bash /net/tscratch/people/plgslukasik/immu-vis/multiplex-image-model/scripts/eval_f256_fourier_pending.sh"
$EVAL_PENDING

chain_run srun torchrun \
    --nnodes=1 \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_masked_model_ddp.py configs/train_equivariant_v3_ddp_zs_hn_f256_fourier_config.yaml
[ -f "$CHAIN_DONE_FILE" ] && $EVAL_PENDING
