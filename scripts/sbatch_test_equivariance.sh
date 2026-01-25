#!/bin/bash
#SBATCH --job-name=mask-imc-v2
#SBATCH --partition=dc-hwai          # or use gpu-enabled queue if available
#SBATCH --account=hai_1191
#SBATCH --ntasks=1
#SBATCH --time=01-00:00:00            # wall time limit (DD-HH:MM:SS format)
#SBATCH --gpus-per-node=1          # new syntax for requesting a GPU per node
#SBATCH --cpus-per-task=4          # number of CPU cores (for dataloader)
#SBATCH --mem=30G                   # total RAM per node
#SBATCH --output=logs/mask_v2_%j.out
#SBATCH --error=logs/mask_v2_%j.err


echo "=========================================="
echo "SBATCH Equivariance Evaluation TEST"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Load modules
module load Stages/2025
module load GCCcore/.13.3.0
module load Python/3.12.3

# Activate virtual environment
source /p/project1/hai_1191/lukasik1/venvs/immu-vis/bin/activate

# Configuration
CONFIG="train_masked_equivariant_config_flip_v2 equiv_tests.yaml"
OUTPUT_DIR="equivariance_test_results_${SLURM_JOB_ID}"
CHECKPOINT=${1:-""}  # Optional checkpoint path as first argument

# Test settings
DEVICE="cpu"
NUM_BATCHES=2
NUM_SAMPLES=2

echo "Config: $CONFIG"
echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Num batches: $NUM_BATCHES (testing mode)"
echo "Num samples: $NUM_SAMPLES"

if [ -n "$CHECKPOINT" ]; then
    echo "Checkpoint: $CHECKPOINT"
    CHECKPOINT_ARG="--checkpoint $CHECKPOINT"
else
    echo "Checkpoint: Using checkpoint from config"
    CHECKPOINT_ARG=""
fi

echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Run evaluation
echo ""
echo "Step 1/2: Running evaluation..."
echo "----------------------------------------"
srun python evaluate_equivariance.py \
    --config "$CONFIG" \
    $CHECKPOINT_ARG \
    --device "$DEVICE" \
    --output-dir "$OUTPUT_DIR" \
    --num-batches "$NUM_BATCHES" \
    --layer-indices -1 \
    --save-features \
    --save-reconstructions

echo ""
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"

# Run visualization
echo ""
echo "Step 2/2: Generating Visualizations"
echo "=========================================="

srun python visualize_equivariance_results.py \
    --results-dir "$OUTPUT_DIR" \
    --batch-idx 0 \
    --sample-idx 0 \
    --num-samples "$NUM_SAMPLES" \
    --rotations 90 180 45 \
    --channel-idx 0

echo ""
echo "=========================================="
echo "TEST COMPLETE!"
echo "=========================================="
echo ""
echo "Results summary:"
echo "  - Results directory: $OUTPUT_DIR"
echo "  - Plots directory: $OUTPUT_DIR/plots"
echo "  - Metrics file: $OUTPUT_DIR/aggregated_metrics.json"
echo "  - Job ID: $SLURM_JOB_ID"
echo ""
echo "To view results:"
echo "  cat $OUTPUT_DIR/aggregated_metrics.json"
echo ""
echo "=========================================="
