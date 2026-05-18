#!/usr/bin/env bash
# Run the equivariant ConvNeXt autoencoder on the cell-typing downstream task.
#
# Uses the existing training venv (already has torch + escnn + sklearn);
# the celltyping uv env is intentionally NOT used (rds2py won't build here,
# and the equivariant path doesn't need VirTues/ESM-2).
#
# Prereq: gold_standard/gs_config.yaml `raw:` paths filled in + the labelled
# nsclc2 data present on disk. Until then, step 1 will fail by design.
set -euo pipefail

VENV=/net/tscratch/people/plgslukasik/venvs/immu-vis/bin/python
PROJ=/net/tscratch/people/plgslukasik/immu-vis/multiplex-image-model/celltyping-downstream/celltyping
CONFIG=./gold_standard/gs_config.yaml
RUN=equiv_convnext_v2_patch

cd "$PROJ"
# celltyping uses `python -m core.*`; make the project importable.
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"

echo "=== 1/3 Build processed dataset from raw labelled data ==="
"$VENV" -m gold_standard.data

echo "=== 2/3 Equivariant embedding inference (VirTues-free runner) ==="
"$VENV" -m core.embeddings.inference_equivariant \
  --config "$CONFIG" \
  --registry ./core/models/registry.yaml \
  --model equiv_convnext_v2 \
  --scheme patch \
  --batch_size 32 \
  --split test

echo "=== 3/3 Cross-validated logistic regression (the downstream metric) ==="
"$VENV" -m core.crossval --config "$CONFIG" --run "$RUN"

echo "Done. Results under: \$base_path/crossval/"
