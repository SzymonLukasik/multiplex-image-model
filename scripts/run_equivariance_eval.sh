#!/bin/bash
#SBATCH --job-name=mask-imc-v2
#SBATCH --partition=dc-hwai          # or use gpu-enabled queue if available
#SBATCH --account=hai_1191
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01-00:00:00            # wall time limit (DD-HH:MM:SS format)
#SBATCH --gpus-per-node=1          # new syntax for requesting a GPU per node
#SBATCH --cpus-per-task=4          # number of CPU cores (for dataloader)
#SBATCH --mem=30G                   # total RAM per node
#SBATCH --output=logs/mask_v2_%j.out
#SBATCH --error=logs/run_eval_%j.err

# module load cuda/11.7             # ensure correct CUDA setup
module load Stages/2025
module load GCCcore/.13.3.0
module load Python/3.12.3
source /p/project1/hai_1191/lukasik1/venvs/immu-vis/bin/activate    # activate your Python environment

# srun python ./train_masked_model.py /home/szlukasik/immu-vis/multiplex-image-model/train_masked_config.yaml

# srun python ./train_masked_model_impainting.py /home/szlukasik/immu-vis/multiplex-image-model/train_masked_equivariant_config_impainting.yaml

# srun python ./train_masked_model_old_loader.py "/home/szlukasik/immu-vis/multiplex-image-model/train_masked_equivariant_config_old_loader.yaml" 

# srun ./run_equivariance_eval.sh

# srun python run_equivariance_eval.py \
#   --config train_masked_equivariant_config_impainting.yaml \
#   --model-checkpoints /raid_encrypted/immucan/models/checskpoint-EquivariantConvnext_20251004_191704_J2206-epoch_179.pth  \
#   --model-type EquivariantConvnext \
#   --split test \
#   --rotation-step 5 \
#   --include-horizontal-flip \
#   --crop-size 113 \
#   --device cuda:0 \
#   --output equivariance_metrics_detailed.json \
#   --seed 42

srun python run_equivariance_eval_notebook.py