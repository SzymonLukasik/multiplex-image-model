#!/bin/bash
#
# Test baseline (non-equivariant) model's rotation behavior
# This demonstrates the improvement from equivariant architecture
#
# Usage:
#   ./scripts/test_baseline_equivariance.sh [checkpoint_path]
#

set -e  # Exit on error

# Environment setup
# Note: Run this script with the environment already activated
# module load Stages/2025
# module load GCCcore/.13.3.0
# module load Python/3.12.3
# source /p/project1/hai_1191/lukasik1/venvs/immu-vis/bin/activate

# Configuration
CONFIG="train_masked_config_baseline.yaml"
OUTPUT_DIR="baseline_equivariance_results_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT=${1:-""}  # Optional checkpoint path as first argument

# Test settings
DEVICE="cpu"
NUM_BATCHES=1  # Small number for testing
NUM_SAMPLES=1  # Number of samples to visualize per batch

echo "=========================================="
echo "Baseline Model Equivariance Test"
echo "=========================================="
echo "Config: $CONFIG"
echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Num batches: $NUM_BATCHES (testing mode)"
echo "Num samples: $NUM_SAMPLES"
echo ""
echo "NOTE: This is a NON-EQUIVARIANT baseline model."
echo "Expect higher rotation errors compared to equivariant model."

if [ -n "$CHECKPOINT" ]; then
    echo "Checkpoint: $CHECKPOINT"
    CHECKPOINT_ARG="--checkpoint $CHECKPOINT"
else
    echo "Checkpoint: Using checkpoint from config"
    CHECKPOINT_ARG=""
fi

echo "=========================================="

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

# Run evaluation
echo ""
echo "Step 1/2: Running evaluation..."
echo "----------------------------------------"
python evaluate_equivariance.py \
    --config "$CONFIG" \
    $CHECKPOINT_ARG \
    --device "$DEVICE" \
    --output-dir "$OUTPUT_DIR" \
    --num-batches "$NUM_BATCHES" \
    --layer-indices -1 \
    --save-reconstructions

echo ""
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"

# Check if evaluation produced output
if [ ! -f "$OUTPUT_DIR/aggregated_metrics.json" ]; then
    echo "ERROR: Evaluation did not produce expected output"
    exit 1
fi

echo ""
echo "Step 2/2: Generating Visualizations"
echo "=========================================="

# Run visualization with multiple samples
python visualize_equivariance_results.py \
    --results-dir "$OUTPUT_DIR" \
    --batch-idx 0 \
    --sample-idx 0 \
    --num-samples "$NUM_SAMPLES" \
    --rotations 90 180 45 \
    --channel-idx 0

echo ""
echo "=========================================="
echo "BASELINE TEST COMPLETE!"
echo "=========================================="
echo ""
echo "Results summary:"
echo "  - Results directory: $OUTPUT_DIR"
echo "  - Plots directory: $OUTPUT_DIR/plots"
echo "  - Metrics file: $OUTPUT_DIR/aggregated_metrics.json"
echo ""

# Display aggregated metrics if available
if command -v jq &> /dev/null; then
    echo "Baseline Model Metrics Preview:"
    echo "----------------------------------------"
    jq '.' "$OUTPUT_DIR/aggregated_metrics.json" 2>/dev/null || cat "$OUTPUT_DIR/aggregated_metrics.json"
else
    echo "Baseline Model Metrics:"
    echo "----------------------------------------"
    cat "$OUTPUT_DIR/aggregated_metrics.json"
fi

echo ""
echo "=========================================="
echo "Next steps:"
echo "  1. Compare with equivariant model results"
echo "  2. Expected: Baseline has MUCH higher rotation errors"
echo "  3. This demonstrates the value of equivariant architecture"
echo "=========================================="
