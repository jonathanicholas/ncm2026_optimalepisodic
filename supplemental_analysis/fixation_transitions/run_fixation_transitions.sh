#!/usr/bin/env bash
# Fixation-transition structure supplementary pipeline (Figure S6).
#
# Produces the transition-matrix and consecutive-sequence figure comparing
# humans with the prior-memory network, plus the accompanying Bayesian
# mixed-effects statistics. Depends on the metarnn "human_like" export, so
# run this AFTER metarnn/run_nn_pipeline.sh.
#
# Usage (from anywhere):
#   bash supplemental_analysis/fixation_transitions/run_fixation_transitions.sh <SIM_NAME> <NINPUTS>
#   bash supplemental_analysis/fixation_transitions/run_fixation_transitions.sh 04_04 5
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."  # cd to repo root

SIM_NAME="${1:-04_04}"
NINPUTS="${2:-5}"
TAG="${SIM_NAME}_input${NINPUTS}"
NN_ROOT="metarnn/simulations/human_like_${TAG}"

if [ ! -d "${NN_ROOT}" ]; then
  echo "[ERROR] NN export not found: ${NN_ROOT}" >&2
  echo "        Run metarnn/run_nn_pipeline.sh first." >&2
  exit 1
fi

mkdir -p output/figures/supplementary

# ── Step 1: Transition / sequence figure ─────────────────────────────────────
echo "[STEP 1/3] Fixation-transition figure"
conda run -n analysis python \
  supplemental_analysis/fixation_transitions/plot_fixation_transitions.py \
  --nn-root "${NN_ROOT}" --tag "${TAG}" --n-sims 1000 --no-show

# ── Step 2: Bayesian mixed-effects statistics ────────────────────────────────
echo "[STEP 2/3] Fixation-transition statistics"
conda run -n analysis Rscript \
  supplemental_analysis/fixation_transitions/run_mixed_effects_fixation_transitions.R \
  --nn-root "${NN_ROOT}" --tag "${TAG}"

# ── Step 3: Copy figure to output/figures/supplementary/ ─────────────────────
echo "[STEP 3/3] Copy figure to output/figures/supplementary/"
cp "${NN_ROOT}/output/next_fixation_gen/FigureFixationTransitions_${TAG}.pdf" \
   output/figures/supplementary/FigureS6.pdf

echo "[DONE] Fixation-transition pipeline complete."
echo "[DONE] Figure S6 -> output/figures/supplementary/FigureS6.pdf"
