#!/usr/bin/env bash
# Reproduce the feature-dimension supplementary figure end-to-end:
#   1. Build choices.csv from the raw subject data.
#   2. Build the two eye-fixation datasets from the main paper pipeline output.
#   3. Fit the eight per-dimension brms models and write the deviation CSV.
#   4. Render FigureS7.pdf from the deviation CSV.
#
# Assumes the `analysis` conda env is available.
# Usage: ./run_feature_analyses.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate the analysis env so python/R see the right packages.
# shellcheck disable=SC1091
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate analysis

echo "[1/4] Building choices.csv..."
python scripts/build_choices.py

echo "[2/4] Building eye-fixation datasets..."
python scripts/build_eye_fixation.py

echo "[3/4] Fitting per-dimension brms models (this takes ~30-40 min)..."
Rscript scripts/compute_feature_stats.R

echo "[4/4] Rendering FigureS7.pdf..."
python scripts/plot_feature_deviation.py

echo "Done. Output: $SCRIPT_DIR/output/FigureS7.pdf"
