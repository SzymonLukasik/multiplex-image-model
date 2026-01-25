#!/bin/bash
#
# Script to run rotation equivariance evaluation
#
# Usage:
#   ./scripts/run_equivariance_evaluation.sh [checkpoint_path]
#

set -e  # Exit on error

# Configuration
CONFIG="train_masked_equivariant_config_flip_v2.yaml"
OUTPUT_DIR="equivariance_results_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT=${1:-""}  # Optional checkpoint path as first argument

# SLURM/GPU settings (adjust as needed)
DEVICE="cpu"
NUM_BATCHES="1"  # Empty means evaluate all batches

echo "=========================================="
echo "Rotation Equivariance Evaluation"
echo "=========================================="
echo "Config: $CONFIG"
echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE"

if [ -n "$CHECKPOINT" ]; then
    echo "Checkpoint: $CHECKPOINT"
    CHECKPOINT_ARG="--checkpoint $CHECKPOINT"
else
    echo "Checkpoint: Using checkpoint from config"
    CHECKPOINT_ARG=""
fi

echo "=========================================="

# Run evaluation
echo "Running evaluation..."
python evaluate_equivariance.py \
    --config "$CONFIG" \
    $CHECKPOINT_ARG \
    --device "$DEVICE" \
    --output-dir "$OUTPUT_DIR" \
    ${NUM_BATCHES:+--num-batches $NUM_BATCHES} \
    --save-features \
    --save-reconstructions \
    --save-latents

echo ""
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"

# Run visualization
echo ""
echo "=========================================="
echo "Generating Visualizations"
echo "=========================================="

python visualize_equivariance_results.py \
    --results-dir "$OUTPUT_DIR" \
    --batch-idx 0 \
    --sample-idx 0 \
    --rotations 30 45 90 135 180 270

echo ""
echo "=========================================="
echo "All done!"
echo "Results: $OUTPUT_DIR"
echo "Plots: $OUTPUT_DIR/plots"
echo "=========================================="
