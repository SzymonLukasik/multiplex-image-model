#!/bin/bash
#
# Compare equivariant and baseline model results
#
# Usage:
#   ./scripts/compare_models.sh equivariant_results_dir baseline_results_dir
#

set -e

EQUIVARIANT_DIR=${1:-""}
BASELINE_DIR=${2:-""}

if [ -z "$EQUIVARIANT_DIR" ] || [ -z "$BASELINE_DIR" ]; then
    echo "Usage: $0 <equivariant_results_dir> <baseline_results_dir>"
    echo ""
    echo "Example:"
    echo "  $0 equivariance_test_results_20260123_042321 baseline_equivariance_results_20260123_123456"
    exit 1
fi

if [ ! -d "$EQUIVARIANT_DIR" ] || [ ! -d "$BASELINE_DIR" ]; then
    echo "ERROR: One or both result directories not found"
    echo "  Equivariant: $EQUIVARIANT_DIR"
    echo "  Baseline: $BASELINE_DIR"
    exit 1
fi

echo "========================================================================"
echo "Model Comparison: Equivariant vs Baseline"
echo "========================================================================"
echo ""
echo "Equivariant Model: $EQUIVARIANT_DIR"
echo "Baseline Model:    $BASELINE_DIR"
echo ""

# Function to extract metric
extract_metric() {
    local file=$1
    local key=$2
    local metric=$3
    python3 -c "import json; data=json.load(open('$file')); print(f\"{data.get('$key', {}).get('$metric', 'N/A'):.6f}\" if isinstance(data.get('$key', {}).get('$metric'), (int, float)) else 'N/A')"
}

echo "=========================================="
echo "90° Rotation Performance"
echo "=========================================="
echo ""
printf "%-20s %-20s %-20s %-20s\n" "Model" "MSE" "L1" "Cosine Sim"
printf "%-20s %-20s %-20s %-20s\n" "--------------------" "--------------------" "--------------------" "--------------------"

eq_mse=$(extract_metric "$EQUIVARIANT_DIR/aggregated_metrics.json" "r90_nf" "equiv_mse_mean")
eq_l1=$(extract_metric "$EQUIVARIANT_DIR/aggregated_metrics.json" "r90_nf" "equiv_l1_mean")
eq_cos=$(extract_metric "$EQUIVARIANT_DIR/aggregated_metrics.json" "r90_nf" "cosine_sim_mean")
printf "%-20s %-20s %-20s %-20s\n" "Equivariant" "$eq_mse" "$eq_l1" "$eq_cos"

bl_mse=$(extract_metric "$BASELINE_DIR/aggregated_metrics.json" "r90_nf" "equiv_mse_mean")
bl_l1=$(extract_metric "$BASELINE_DIR/aggregated_metrics.json" "r90_nf" "equiv_l1_mean")
bl_cos=$(extract_metric "$BASELINE_DIR/aggregated_metrics.json" "r90_nf" "cosine_sim_mean")
printf "%-20s %-20s %-20s %-20s\n" "Baseline" "$bl_mse" "$bl_l1" "$bl_cos"

echo ""
echo "=========================================="
echo "180° Rotation Performance"
echo "=========================================="
echo ""
printf "%-20s %-20s %-20s %-20s\n" "Model" "MSE" "L1" "Cosine Sim"
printf "%-20s %-20s %-20s %-20s\n" "--------------------" "--------------------" "--------------------" "--------------------"

eq_mse=$(extract_metric "$EQUIVARIANT_DIR/aggregated_metrics.json" "r180_nf" "equiv_mse_mean")
eq_l1=$(extract_metric "$EQUIVARIANT_DIR/aggregated_metrics.json" "r180_nf" "equiv_l1_mean")
eq_cos=$(extract_metric "$EQUIVARIANT_DIR/aggregated_metrics.json" "r180_nf" "cosine_sim_mean")
printf "%-20s %-20s %-20s %-20s\n" "Equivariant" "$eq_mse" "$eq_l1" "$eq_cos"

bl_mse=$(extract_metric "$BASELINE_DIR/aggregated_metrics.json" "r180_nf" "equiv_mse_mean")
bl_l1=$(extract_metric "$BASELINE_DIR/aggregated_metrics.json" "r180_nf" "equiv_l1_mean")
bl_cos=$(extract_metric "$BASELINE_DIR/aggregated_metrics.json" "r180_nf" "cosine_sim_mean")
printf "%-20s %-20s %-20s %-20s\n" "Baseline" "$bl_mse" "$bl_l1" "$bl_cos"

echo ""
echo "=========================================="
echo "45° Rotation Performance"
echo "=========================================="
echo ""
printf "%-20s %-20s %-20s %-20s\n" "Model" "MSE" "L1" "Cosine Sim"
printf "%-20s %-20s %-20s %-20s\n" "--------------------" "--------------------" "--------------------" "--------------------"

eq_mse=$(extract_metric "$EQUIVARIANT_DIR/aggregated_metrics.json" "r45_nf" "equiv_mse_mean")
eq_l1=$(extract_metric "$EQUIVARIANT_DIR/aggregated_metrics.json" "r45_nf" "equiv_l1_mean")
eq_cos=$(extract_metric "$EQUIVARIANT_DIR/aggregated_metrics.json" "r45_nf" "cosine_sim_mean")
printf "%-20s %-20s %-20s %-20s\n" "Equivariant" "$eq_mse" "$eq_l1" "$eq_cos"

bl_mse=$(extract_metric "$BASELINE_DIR/aggregated_metrics.json" "r45_nf" "equiv_mse_mean")
bl_l1=$(extract_metric "$BASELINE_DIR/aggregated_metrics.json" "r45_nf" "equiv_l1_mean")
bl_cos=$(extract_metric "$BASELINE_DIR/aggregated_metrics.json" "r45_nf" "cosine_sim_mean")
printf "%-20s %-20s %-20s %-20s\n" "Baseline" "$bl_mse" "$bl_l1" "$bl_cos"

echo ""
echo "========================================================================"
echo "Summary"
echo "========================================================================"
echo ""
echo "✅ Lower MSE/L1 = Better equivariance"
echo "✅ Higher Cosine Similarity = Better equivariance"
echo ""
echo "Expected: Equivariant model should show:"
echo "  - Much lower MSE/L1 (ideally < 0.01 for 90° rotations)"
echo "  - Much higher Cosine Similarity (ideally > 0.99)"
echo ""
echo "This demonstrates the value of equivariant architecture!"
echo "========================================================================"
