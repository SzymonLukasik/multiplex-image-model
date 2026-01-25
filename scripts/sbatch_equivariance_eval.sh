#!/bin/bash
#SBATCH --job-name=equiv-eval
#SBATCH --partition=dc-gpu
#SBATCH --account=hai_1191
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/equiv_eval_%j.out
#SBATCH --error=logs/equiv_eval_%j.err

# Full Equivariance Evaluation (all test batches, GPU)

echo "=========================================="
echo "SBATCH Equivariance Evaluation (FULL)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Load modules
module load Stages/2025
module load GCCcore/.13.3.0
module load Python/3.12.3
module load CUDA/12.1.0  # Adjust CUDA version as needed

# Activate virtual environment
source /p/project1/hai_1191/lukasik1/venvs/immu-vis/bin/activate

# Configuration
CONFIG=${1:-"train_masked_equivariant_config_flip_v2.yaml"}
CHECKPOINT=${2:-""}  # Optional checkpoint path
OUTPUT_DIR="equivariance_results_${SLURM_JOB_ID}"

# Settings
DEVICE="cuda"
NUM_BATCHES=""  # Empty = all batches
NUM_SAMPLES=3   # Visualize 3 samples per batch
LAYER_INDICES="-1"  # Evaluate only last (trivial) layer

echo "Config: $CONFIG"
echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Num batches: ${NUM_BATCHES:-all}"
echo "Num samples: $NUM_SAMPLES"
echo "Layer indices: $LAYER_INDICES"

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
    ${NUM_BATCHES:+--num-batches $NUM_BATCHES} \
    --layer-indices $LAYER_INDICES \
    --save-features \
    --save-reconstructions

echo ""
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"

# Check if evaluation produced output
if [ ! -f "$OUTPUT_DIR/aggregated_metrics.json" ]; then
    echo "ERROR: Evaluation did not produce expected output"
    exit 1
fi

# Run visualization
echo ""
echo "Step 2/2: Generating Visualizations"
echo "=========================================="

srun python visualize_equivariance_results.py \
    --results-dir "$OUTPUT_DIR" \
    --batch-idx 0 \
    --sample-idx 0 \
    --num-samples "$NUM_SAMPLES" \
    --rotations 90 180 270 45 30 \
    --channel-idx 0

echo ""
echo "=========================================="
echo "EVALUATION COMPLETE!"
echo "=========================================="
echo ""
echo "Results summary:"
echo "  - Results directory: $OUTPUT_DIR"
echo "  - Plots directory: $OUTPUT_DIR/plots"
echo "  - Metrics file: $OUTPUT_DIR/aggregated_metrics.json"
echo "  - Job ID: $SLURM_JOB_ID"
echo ""

# Display aggregated metrics if available
if [ -f "$OUTPUT_DIR/aggregated_metrics.json" ]; then
    echo "Aggregated Metrics Preview:"
    echo "----------------------------------------"
    head -50 "$OUTPUT_DIR/aggregated_metrics.json"
fi

echo ""
echo "=========================================="
echo "To view full results:"
echo "  cat $OUTPUT_DIR/aggregated_metrics.json"
echo "  ls $OUTPUT_DIR/plots/"
echo ""
echo "To download results:"
echo "  scp -r <user>@jrlogin.fz-juelich.de:$(pwd)/$OUTPUT_DIR ."
echo "=========================================="
