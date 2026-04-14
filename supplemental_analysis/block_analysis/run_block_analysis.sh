#!/usr/bin/env bash
# Reproduce the block-number interaction table end-to-end:
#   1. Build choices.csv from the raw subject data.
#   2. Build the two eye-fixation datasets from the main paper pipeline output.
#   3. Fit the eight brms models and write the interaction table.
#
# Assumes the `analysis` conda env is available.
# Usage: ./run_block_analysis.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# shellcheck disable=SC1091
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate analysis

echo "[1/3] Building choices.csv..."
python scripts/build_choices.py

echo "[2/3] Building eye-fixation datasets..."
python scripts/build_eye_fixation.py

echo "[3/3] Fitting block-interaction brms models (~30-45 min)..."
Rscript scripts/compute_block_stats.R

echo "Done. Output: $SCRIPT_DIR/output/block_interaction_table.csv"
