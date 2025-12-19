#!/bin/bash
#SBATCH --job-name=mask-imc-v2
#SBATCH --partition=common          # or use gpu-enabled queue if available
#SBATCH --qos=szlukasik             # required per cluster policy
#SBATCH --time=07-00:00:00            # wall time limit (DD-HH:MM:SS format)
#SBATCH --gpus-per-node=1          # new syntax for requesting a GPU per node
#SBATCH --cpus-per-task=8          # number of CPU cores (for dataloader)
#SBATCH --mem=100G                   # total RAM per node
#SBATCH --output=logs/mask_v2_%j.out
#SBATCH --error=logs/mask_v2_%j.err

# module load cuda/11.7             # ensure correct CUDA setup
source ~/venvs/immu-vis/bin/activate    # activate your Python environment

# srun python ./train_masked_model.py /home/szlukasik/immu-vis/multiplex-image-model/train_masked_config.yaml

srun python ./train_masked_model_impainting.py /home/szlukasik/immu-vis/multiplex-image-model/train_masked_equivariant_config_flip_v2.yaml

# srun cp /raid_encrypted/immucan/models/checkpoint-EquivariantConvnext_20251004_191704_J2206-epoch_179.pth /home/szlukasik/immu-vis/multiplex-image-model/checkpoints/



# srun python ./train_masked_model_old_loader.py "/home/szlukasik/immu-vis/multiplex-image-model/train_masked_equivariant_config_old_loader.yaml" 
