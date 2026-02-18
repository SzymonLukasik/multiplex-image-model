#!/bin/bash
#SBATCH --job-name=equiv-eval-v2
#SBATCH --partition=dc-hwai
#SBATCH --account=hai_1191
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00-06:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --output=logs/equiv_eval_v2_%j.out
#SBATCH --error=logs/equiv_eval_v2_%j.err

module load Stages/2025
module load GCCcore/.13.3.0
module load Python/3.12.3

source /p/project1/hai_1191/lukasik1/venvs/immu-vis/bin/activate

cd /p/project1/hai_1191/lukasik1/immu-vis/multiplex-image-model

srun python ./evaluate_equivariance_v2.py \
    --config train_masked_equivariant_config_flip_v2.yaml \
    --output-dir equivariance_results_v2

# Alternative baseline config:
# srun python ./evaluate_equivariance_v2.py \
#     --config train_masked_config.yaml \
#     --output-dir equivariance_results_v2
