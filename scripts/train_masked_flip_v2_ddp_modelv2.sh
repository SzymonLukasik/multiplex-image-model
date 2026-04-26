#!/bin/bash
#SBATCH --account=plgspacelet2-gpu-a100
#SBATCH --job-name=mask-imc-v2-ddp-modelv2
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=02-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # one srun task; torchrun spawns the GPU processes
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16           # 8 CPUs per GPU × 2 GPUs
#SBATCH --mem=200G
#SBATCH --output=logs/mask_v2_ddp_modelv2_%j.out
#SBATCH --error=logs/mask_v2_ddp_modelv2_%j.err

source $SCRATCH/venvs/immu-vis/bin/activate

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)

srun torchrun \
    --nnodes=1 \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    ./train_masked_model_ddp_v2.py \
    $SCRATCH/immu-vis/multiplex-image-model/train_masked_equivariant_config_flip_v2_wider_modelv2_1.yaml
