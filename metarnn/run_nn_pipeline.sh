#!/usr/bin/env bash
# Master script for the full NN processing pipeline.
# Processes both input0 (baseline) and inputN, generates all figures,
# runs NN-NN comparison, and fits Bayesian mixed-effects models.
#
# All commands assume they are run from the repository root directory.
#
# Usage:
#   bash metarnn/run_nn_pipeline.sh <SIM_NAME> <NINPUTS>
#   bash metarnn/run_nn_pipeline.sh 04_04 5
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."  # cd to repo root

# ── Parse arguments ──────────────────────────────────────────────────────────
SIM_NAME="${1:?Usage: $0 <SIM_NAME> <NINPUTS>  (e.g. 04_04 5)}"
NINPUTS="${2:?Usage: $0 <SIM_NAME> <NINPUTS>  (e.g. 04_04 5)}"

TAG0="${SIM_NAME}_input0"
TAGN="${SIM_NAME}_input${NINPUTS}"

SIM_DIR0="metarnn/simulations/simulation_${TAG0}"
SIM_DIRN="metarnn/simulations/simulation_${TAGN}"

OUT0="metarnn/simulations/human_like_${TAG0}"
OUTN="metarnn/simulations/human_like_${TAGN}"

echo "============================================================"
echo "[INFO] NN Pipeline: ${SIM_NAME}, comparing input0 vs input${NINPUTS}"
echo "[INFO] Baseline:  ${SIM_DIR0} -> ${OUT0}"
echo "[INFO] Target:    ${SIM_DIRN} -> ${OUTN}"
echo "============================================================"

# ── Step 1: Process input0 (baseline) ────────────────────────────────────────
echo ""
if [ -d "${OUT0}/output" ]; then
  echo "[STEP 1/8] SKIP — input0 baseline already exists at ${OUT0}"
else
  echo "[STEP 1/8] Process input0 baseline (create_nn_figures.sh)"
  bash metarnn/create_nn_figures.sh \
    --sim-dir "${SIM_DIR0}" --out-root "${OUT0}" --tag "${TAG0}"
fi

# ── Step 2: Process inputN ───────────────────────────────────────────────────
echo ""
echo "[STEP 2/8] Process input${NINPUTS} (create_nn_figures.sh)"
bash metarnn/create_nn_figures.sh \
  --sim-dir "${SIM_DIRN}" --out-root "${OUTN}" --tag "${TAGN}"

# ── Step 3: CV accuracy comparison (human vs NN with prop-recall drop) ────────
echo ""
echo "[STEP 3/9] Compare human vs NN CV accuracy (prop-recall drop)"
NN_DROP_CSV="${OUTN}/output/eyegaze/stats/droprecall/summary_prop_time_location_interactions_true_all_norm-within.csv"
HUMAN_CV_CSV="output/eyegaze/stats/summary_prop_time_location_interactions_recalled_all_norm-within.csv"
if [ -f "${NN_DROP_CSV}" ]; then
  conda run -n analysis python metarnn/lib/compare_cv_accuracy_human_vs_nn.py \
    --nn-drop-csv "${NN_DROP_CSV}" \
    --human-csv "${HUMAN_CV_CSV}" \
    --out-dir "${OUTN}/output/human_vs_nn_brms"
else
  echo "  SKIP — drop-recall summary not found: ${NN_DROP_CSV}"
fi

# ── Step 4: Human vs NN R stats (for inputN) ────────────────────────────────
echo ""
echo "[STEP 4/9] Human vs NN Bayesian mixed-effects models (inputN)"
conda run -n analysis Rscript metarnn/run_mixed_effects_human_vs_nn.R \
  --nn-root "${OUTN}" \
  --tag "${TAGN}" \
  --out-dir "${OUTN}/output/human_vs_nn_brms"

# ── Step 5: NN vs NN comparison figure ───────────────────────────────────────
echo ""
echo "[STEP 5/9] NN vs NN comparison figure (input0 vs input${NINPUTS})"
conda run -n analysis python metarnn/plot_NN_NN_comparison.py \
  --nn-root1 "${OUTN}" \
  --nn-root2 "${OUT0}" \
  --label1 "Prior Mem." \
  --label2 "No Prior Mem." \
  --tag "${SIM_NAME}_input${NINPUTS}_vs_input0" \
  --nn-tag1 "${TAGN}" \
  --nn-tag2 "${TAG0}"

# ── Step 6: Export NN-NN comparison data for R ───────────────────────────────
echo ""
echo "[STEP 6/9] Export NN-NN comparison panel data for R stats"
conda run -n analysis python metarnn/lib/export_nn_nn_comparison_data.py \
  --nn-root "${OUTN}" \
  --tag "${TAGN}" \
  --out-dir "${OUTN}/output/human_vs_nn_brms/data"

# ── Step 7: NN vs NN comparison R stats ──────────────────────────────────────
echo ""
echo "[STEP 7/9] NN vs NN Bayesian mixed-effects models"
conda run -n analysis Rscript metarnn/run_mixed_effects_nn_nn_comparison.R \
  --data-dir "${OUTN}/output/human_vs_nn_brms/data" \
  --tag "${TAGN}" \
  --out-dir "${OUTN}/output/human_vs_nn_brms"

# ── Step 8: Next-fixation-generation figure ──────────────────────────────────
echo ""
echo "[STEP 8/9] Human vs NN next-fixation-generation figure (inputN)"
conda run -n analysis python metarnn/plot_NN_H_next_fixation_gen.py \
  --nn-root "${OUTN}" \
  --tag "${TAGN}" \
  --n-sims 1000 \
  --no-show

# ── Step 9: Next-fixation-generation R stats ─────────────────────────────────
echo ""
echo "[STEP 9/9] Next-fixation-generation Bayesian mixed-effects models (inputN)"
conda run -n analysis Rscript metarnn/run_mixed_effects_next_fixation_gen.R \
  --nn-root "${OUTN}" \
  --tag "${TAGN}"

# ── Step 10: Evidence accumulation + belief decoding figure ──────────────────
echo ""
echo "[STEP 10/11] Evidence accumulation + belief decoding figure"
EVIDENCE_DIR="${OUT0}/output/evidence"
HIDDEN_DIR="${SIM_DIR0}/with_hidden"
DECODING_CSV="${EVIDENCE_DIR}/belief_decoding_results.csv"

mkdir -p "${EVIDENCE_DIR}"

# Run decoding if CSV doesn't exist and hidden-state JSONs are available
if [ ! -f "${DECODING_CSV}" ] && [ -d "${HIDDEN_DIR}" ]; then
  echo "  Running belief decoding..."
  conda run -n analysis python metarnn/lib/run_belief_decoding.py \
    --data-dir "${HIDDEN_DIR}" --out-dir "${EVIDENCE_DIR}" \
    --prefix data_0 --seeds 5 6 7 8 9 --workers 5
elif [ -f "${DECODING_CSV}" ]; then
  echo "  SKIP decoding — cached CSV exists: ${DECODING_CSV}"
else
  echo "  SKIP decoding — no hidden-state JSONs and no cached CSV"
fi

# Generate figure (from JSONs if available, otherwise from cache)
if [ -d "${HIDDEN_DIR}" ]; then
  conda run -n analysis python metarnn/lib/plot_evidence_figure.py \
    --data-dir "${HIDDEN_DIR}" --out-dir "${EVIDENCE_DIR}" \
    --prefix data_0 --seeds 5 6 7 8 9 \
    --decoding-csv "${DECODING_CSV}" --save-cache
elif [ -f "${EVIDENCE_DIR}/evidence_figure_cache.csv" ]; then
  conda run -n analysis python metarnn/lib/plot_evidence_figure.py \
    --cache-dir "${EVIDENCE_DIR}" --out-dir "${EVIDENCE_DIR}" \
    --decoding-csv "${DECODING_CSV}"
else
  echo "  SKIP evidence figure — no data available"
fi

# ── Step 11: Copy figures to output/figures/ ─────────────────────────────────
echo ""
echo "[STEP 11/11] Copy figures to output/figures/"
mkdir -p output/figures/supplementary

cp "${OUTN}/output/overview/FigureNN_overview_${TAGN}.pdf" \
   output/figures/Figure3.pdf
cp "${OUTN}/output/human_comparison/FigureNN_NN_comparison_${TAGN}_vs_input0.pdf" \
   output/figures/Figure4.pdf
cp "${OUTN}/output/next_fixation_gen/FigureNN_H_next_fixation_gen_${TAGN}.pdf" \
   output/figures/Figure5.pdf

if [ -f "${OUT0}/output/evidence/belief_decoding_supplement.pdf" ]; then
  cp "${OUT0}/output/evidence/belief_decoding_supplement.pdf" \
     output/figures/supplementary/FigureS3.pdf
fi
cp "${OUTN}/output/overview/propDropSupplement_${TAGN}.pdf" \
   output/figures/supplementary/FigureS4.pdf
cp "${OUTN}/output/next_fixation_gen/FigureAdvantageSupplement_${TAGN}.pdf" \
   output/figures/supplementary/FigureS5.pdf
cp "${OUTN}/output/next_fixation_gen/FigureTransitionSupplement_${TAGN}.pdf" \
   output/figures/supplementary/FigureS6.pdf

if [ -f "${OUT0}/output/evidence/evidence_figure.pdf" ]; then
  cp "${OUT0}/output/evidence/evidence_figure.pdf" \
     output/figures/Figure3BC.pdf
fi

echo ""
echo "============================================================"
echo "[DONE] NN pipeline complete for ${SIM_NAME} (input0 vs input${NINPUTS}) [11 steps]"
echo "[DONE] Outputs: ${OUTN}/output/"
echo "[DONE] Figures: output/figures/"
echo "============================================================"
