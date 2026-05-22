#!/usr/bin/env bash
# Next-fixation conditional-logit pipeline.
#
# Builds the long-form candidate datasets, fits the conditional-logit models
# (hierarchical for humans, fixed-effects for the network and null oracles),
# and produces Figure 5 (next-fixation forest) and Figure S5 (null-oracle
# forest). MCMC fits are slow; each step is skipped if its output exists.
#
# Usage (from anywhere):
#   bash metarnn/next_fixation/run_next_fixation.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."  # cd to repo root

OUT_DIR="output/next_fixation"
mkdir -p "${OUT_DIR}" output/figures/supplementary

# ── Step 1: Build long-form candidate datasets ───────────────────────────────
echo "[STEP 1/4] Build next-fixation candidate datasets"
if [ -f "${OUT_DIR}/next_fixation_long_human.csv" ] \
   && [ -f "${OUT_DIR}/next_fixation_long_rnn_input5_500k.csv" ] \
   && [ -f "${OUT_DIR}/next_fixation_long_walk_ring_noisy_10x.csv" ] \
   && [ -f "${OUT_DIR}/next_fixation_long_random_10x.csv" ]; then
  echo "  SKIP — all four long-form CSVs already exist"
else
  conda run -n analysis python metarnn/next_fixation/build_next_fixation_data.py
fi

# ── Step 2: Fit conditional-logit models ─────────────────────────────────────
echo "[STEP 2/4] Fit conditional-logit models"

if [ -f "${OUT_DIR}/conditional_logit_human_population_beta.csv" ]; then
  echo "  SKIP human (hierarchical) — beta CSV exists"
else
  echo "  Fitting human (hierarchical) ..."
  conda run -n analysis Rscript metarnn/next_fixation/fit_conditional_logit_re.R
fi

for DATASET in rnn_input5_500k walk_ring_noisy_10x random_10x; do
  if [ -f "${OUT_DIR}/conditional_logit_${DATASET}_beta.csv" ]; then
    echo "  SKIP ${DATASET} — beta CSV exists"
  else
    echo "  Fitting ${DATASET} ..."
    conda run -n analysis Rscript metarnn/next_fixation/fit_conditional_logit.R "${DATASET}"
  fi
done

# ── Step 3: Generate figures ─────────────────────────────────────────────────
echo "[STEP 3/4] Generate next-fixation figures"
conda run -n analysis python metarnn/next_fixation/plot_next_fixation_forest.py
conda run -n analysis python metarnn/next_fixation/plot_next_fixation_nulls_forest.py

# ── Step 4: Copy figures to output/figures/ ──────────────────────────────────
echo "[STEP 4/4] Copy figures to output/figures/"
cp "${OUT_DIR}/next_fixation_forest.pdf"       output/figures/Figure5.pdf
cp "${OUT_DIR}/next_fixation_nulls_forest.pdf" output/figures/supplementary/FigureS5.pdf

echo "[DONE] Next-fixation pipeline complete."
echo "[DONE] Figure 5  -> output/figures/Figure5.pdf"
echo "[DONE] Figure S5 -> output/figures/supplementary/FigureS5.pdf"
