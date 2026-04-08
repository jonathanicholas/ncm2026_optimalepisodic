#!/bin/bash
# Run the full behavioral analysis pipeline.
#
# Pipeline:
#   1. analyze_behavior.py  — compile behavioral data, generate Figure1.pdf
#   2. run_mixed_effects.R  — fit Bayesian mixed-effects models (brms)
#
# Outputs:
#   output/behavior/Figure1.pdf
#   output/behavior/stats/*.csv  (data + model summaries)
#
# Usage:
#   conda activate analysis
#   bash analysis/run_behavior.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$BASE_DIR"

echo "=== Step 1/2: Compile behavioral data and generate Figure1 ==="
python analysis/analyze_behavior.py
echo "[DONE] analyze_behavior.py"

echo ""
echo "=== Step 2/2: Fit mixed-effects models ==="
Rscript analysis/run_mixed_effects.R
echo "[DONE] run_mixed_effects.R"

echo ""
echo "=== Behavioral pipeline complete ==="
echo "  Figure:  output/behavior/Figure1.pdf"
echo "  Stats:   output/behavior/stats/"
