 #!/usr/bin/env bash
set -euo pipefail

# Create NN figures from an existing NN "human_like" export folder.
#
# Runs all NN figure-generation scripts in sequence.
#
# Usage:
#   bash metarnn/create_nn_figures.sh \
#     --sim-dir metarnn/simulations/simulation_04_04_input5 \
#     --out-root metarnn/simulations/human_like_04_04_input5 \
#     --tag 04_04_input5

OUT_ROOT="metarnn/simulations/human_like"
TAG=""
METRIC="duration"
SIM_DIR=""
JSONS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sim-dir)
      SIM_DIR="$2"
      shift 2
      ;;
    --out-root)
      OUT_ROOT="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --metric)
      METRIC="$2"
      shift 2
      ;;
    --json)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        JSONS+=("$1")
        shift
      done
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${SIM_DIR}" ]]; then
  if [[ ! -d "${SIM_DIR}" ]]; then
    echo "[ERROR] --sim-dir is not a directory: ${SIM_DIR}" >&2
    exit 1
  fi

  while IFS= read -r -d '' f; do
    JSONS+=("$f")
  done < <(find "${SIM_DIR}" -maxdepth 1 -type f -name "*.json" -print0 | sort -z)
fi

if [[ ! -d "${OUT_ROOT}" ]]; then
  # Allow bootstrapping a new OUT_ROOT via compilation.
  mkdir -p "${OUT_ROOT}"
fi

if [[ -z "${TAG}" ]]; then
  TAG="$(basename "${OUT_ROOT}")"
fi

# Output subdirectories under OUT_ROOT/output/
OVERVIEW_DIR="${OUT_ROOT}/output/overview"
HCOMP_DIR="${OUT_ROOT}/output/human_comparison"
NEXT_FIX_DIR="${OUT_ROOT}/output/next_fixation_gen"
NN_EYEGAZE_STATS="${OUT_ROOT}/output/eyegaze/stats"

PRED_SUMMARY_CSV="${NN_EYEGAZE_STATS}/summary_prop_time_location_interactions_true_all_norm-within.csv"

echo "[INFO] Out root: ${OUT_ROOT}"
echo "[INFO] Tag:      ${TAG}"
echo "[INFO] Metric:   ${METRIC}"

have_compiled_outputs() {
  # Need at least one subject with both behavior and fixations (both in data/).
  local any_log
  local any_fix
  any_log="$(find "${OUT_ROOT}/data" -maxdepth 2 -type f -name "*_MAIN_logfile_7.csv" 2>/dev/null | head -n 1 || true)"
  any_fix="$(find "${OUT_ROOT}/data" -maxdepth 2 -type f \( -name "*_fixations_df_original_buffer_50.csv" -o -name "*_fixations_df_original.csv" \) 2>/dev/null | head -n 1 || true)"
  [[ -n "${any_log}" && -n "${any_fix}" ]]
}

have_clean_choice_fixations() {
  local any
  any="$(find "${OUT_ROOT}/output" -maxdepth 1 -type f -name "choice_fixations_clean*.csv" 2>/dev/null | head -n 1 || true)"
  [[ -n "${any}" ]]
}

have_fixations_for_modeling() {
  local any
  any="$(find "${OUT_ROOT}/data" -maxdepth 2 -type f -name "*_fixations_for_modeling*.csv" 2>/dev/null | head -n 1 || true)"
  [[ -n "${any}" ]]
}

have_human_clean_choice_fixations() {
  local any
  any="$(find "${REPO_ROOT}/output" -maxdepth 1 -type f -name "choice_fixations_clean*.csv" 2>/dev/null | head -n 1 || true)"
  [[ -n "${any}" ]]
}

have_human_fixations_for_modeling() {
  local any
  # Fixation-for-modeling CSVs live under data/{SUBID}/
  any="$(find "${REPO_ROOT}/data" -maxdepth 2 -type f -name "*_fixations_for_modeling*.csv" 2>/dev/null | head -n 1 || true)"
  [[ -n "${any}" ]]
}

have_choice_fixation_proportions() {
  local any
  any="$(find "${NN_EYEGAZE_STATS}" -maxdepth 1 -type f -name "choice_fixation_*_subject_means_*.csv" 2>/dev/null | head -n 1 || true)"
  [[ -n "${any}" ]]
}

have_human_vs_nn_revisits_tables() {
  local h
  local n
  h="${REPO_ROOT}/output/eyegaze/stats/revisits_count_and_duration_by_subject_human.csv"
  n="${NN_EYEGAZE_STATS}/revisits_count_and_duration_by_subject_nn_${TAG}.csv"
  [[ -f "${h}" && -f "${n}" ]]
}

echo "[STEP] Ensure human_like export exists (compile JSON -> human-like CSVs if needed)"
if have_compiled_outputs; then
  echo "[INFO] Found compiled data under ${OUT_ROOT}/data; skipping compilation."
else
  if [[ ${#JSONS[@]} -eq 0 ]]; then
    echo "[ERROR] No compiled outputs found under --out-root, and no --json/--sim-dir provided to compile them." >&2
    echo "        Provide either: --sim-dir <dir_with_jsons> OR --json <a.json> <b.json> ..." >&2
    exit 1
  fi
  conda run -n analysis python metarnn/lib/compile_nn_to_human_fixations.py \
    --out_root "${OUT_ROOT}" \
    --json "${JSONS[@]}"
fi

echo "[STEP] Ensure clean choice fixations exist (prepare_choice_fixations.py if needed)"
if have_clean_choice_fixations; then
  echo "[INFO] Found cached clean choice fixations under ${OUT_ROOT}/output; skipping."
else
  conda run -n analysis python analysis/lib/prepare_choice_fixations.py \
    --base-dir "${OUT_ROOT}"
fi

echo "[STEP] Ensure fixations_for_modeling exist (modeling.prepare_fixations_for_modeling if needed)"
if have_fixations_for_modeling; then
  echo "[INFO] Found cached fixations_for_modeling under ${OUT_ROOT}/data; skipping."
else
  conda run -n analysis env PYTHONPATH="${REPO_ROOT}" python -m addm.lib.prepare_fixations_for_modeling \
    --output-dir "${OUT_ROOT}/data" \
    --input-dir "${OUT_ROOT}/data"
fi

echo "[STEP] Ensure choice fixation proportions exist (choice_fixation_proportions.py if needed)"
if have_choice_fixation_proportions; then
  echo "[INFO] Found cached choice_fixation_proportions under ${NN_EYEGAZE_STATS}; skipping."
else
  conda run -n analysis python analysis/lib/choice_fixation_proportions.py \
    --root "${OUT_ROOT}" \
    --metric "${METRIC}"
fi

echo "[STEP] Choice prediction from prop-time interactions (cached)"
if [[ -f "${PRED_SUMMARY_CSV}" ]]; then
  echo "[INFO] Found cached summary: ${PRED_SUMMARY_CSV}"
else
  conda run -n analysis python analysis/lib/predict_choice_from_item_prop_time_interactions.py \
    --root "${OUT_ROOT}" \
    --out-dir "${NN_EYEGAZE_STATS}" \
    --value-source true \
    --feature-set location_interactions \
    --visit-type all \
    --n-bootstrap 50 \
    --n-sims 0
fi

echo "[STEP] Humans vs NN revisits metrics (cached)"
echo "[STEP] Ensure human clean choice fixations exist (prepare_choice_fixations.py if needed)"
if have_human_clean_choice_fixations; then
  echo "[INFO] Found cached human choice_fixations_clean*.csv under ${REPO_ROOT}/output; skipping."
else
  conda run -n analysis python analysis/lib/prepare_choice_fixations.py \
    --base-dir "${REPO_ROOT}"
fi

echo "[STEP] Ensure human fixations_for_modeling exist (modeling.prepare_fixations_for_modeling if needed)"
if have_human_fixations_for_modeling; then
  echo "[INFO] Found cached human fixations_for_modeling under ${REPO_ROOT}/data; skipping."
else
  conda run -n analysis env PYTHONPATH="${REPO_ROOT}" python -m addm.lib.prepare_fixations_for_modeling \
    --output-dir "${REPO_ROOT}/data" \
    --input-dir "${REPO_ROOT}/data"
fi

if have_human_vs_nn_revisits_tables; then
  echo "[INFO] Found cached revisits by-subject tables; skipping."
else
  conda run -n analysis python analysis/lib/compute_revisits_count_and_duration.py \
    --base-dir "${REPO_ROOT}" \
    --out-dir "${REPO_ROOT}/output/eyegaze/stats" \
    --output-tag "${TAG}" \
    --nn-base-dir "${OUT_ROOT}" \
    --nn-out-dir "${NN_EYEGAZE_STATS}" \
    --nn-exclude-subjects \
    --nn-time-unit-label steps
fi

HUMAN_PRED_DIR="${REPO_ROOT}/output/eyegaze/stats"
HUMAN_PRED_SUMMARY="${HUMAN_PRED_DIR}/summary_prop_time_location_interactions_recalled_all_norm-within.csv"

echo "[STEP] Human choice prediction from prop-time interactions (cached)"
if [[ -f "${HUMAN_PRED_SUMMARY}" ]]; then
  echo "[INFO] Found cached human summary: ${HUMAN_PRED_SUMMARY}"
else
  conda run -n analysis python analysis/lib/predict_choice_from_item_prop_time_interactions.py \
    --out-dir "${HUMAN_PRED_DIR}" \
    --value-source recalled \
    --feature-set location_interactions \
    --visit-type all \
    --n-bootstrap 1000 \
    --n-sims 0
fi

echo "[STEP] Compute recall drop fraction (cached)"
RECALL_SIG_CSV="${REPO_ROOT}/output/eyegaze/recall/recall_sig_timepoint_prop_fix_time.csv"
if [[ -f "${RECALL_SIG_CSV}" ]]; then
  echo "[INFO] Found cached recall sig CSV: ${RECALL_SIG_CSV}"
else
  conda run -n analysis python analysis/lib/compute_recall_drop_fraction.py \
    --data-dir "${REPO_ROOT}/data" \
    --group-csv "${REPO_ROOT}/output/eyegaze/recall/group_time_course_original_buffer_50.csv" \
    --out-csv "${RECALL_SIG_CSV}"
fi

echo "[STEP] Choice prediction with recall-calibrated fixation drop (droprecall; cached)"
NN_DROPRECALL_DIR="${NN_EYEGAZE_STATS}/droprecall"
DROPRECALL_SUMMARY="${NN_DROPRECALL_DIR}/summary_prop_time_location_interactions_true_all_norm-within.csv"
if [[ -f "${DROPRECALL_SUMMARY}" ]]; then
  echo "[INFO] Found cached droprecall summary: ${DROPRECALL_SUMMARY}"
else
  DROP_FRAC=$(python3 -c "
import csv
with open('${RECALL_SIG_CSV}') as f:
    r = next(csv.DictReader(f))
    print(f'{1 - float(r[\"mean_prop_fix_time_sig\"]):.6f}')
")
  echo "[INFO] Using drop_frac=${DROP_FRAC}"
  conda run -n analysis python analysis/lib/predict_choice_from_item_prop_time_interactions.py \
    --root "${OUT_ROOT}" \
    --out-dir "${NN_DROPRECALL_DIR}" \
    --value-source true \
    --feature-set location_interactions \
    --visit-type all \
    --n-bootstrap 50 \
    --n-sims 0 \
    --drop-frac "${DROP_FRAC}"
fi

echo "[STEP] Drop-fraction sweep for prop-drop supplement (cached)"
DROP_SWEEP_CSV="${NN_EYEGAZE_STATS}/drop_sweep_summary.csv"
if [[ -f "${DROP_SWEEP_CSV}" ]]; then
  echo "[INFO] Found cached drop sweep: ${DROP_SWEEP_CSV}"
else
  DROP_SWEEP_TMP="${NN_EYEGAZE_STATS}/drop_sweep_tmp"
  mkdir -p "${DROP_SWEEP_TMP}"
  for PCT in 10 20 30 40 50 60 70 80 90; do
    FRAC=$(python3 -c "print(f'${PCT}/100:.2f}'.format())" 2>/dev/null || python3 -c "print(${PCT}/100)")
    TMP_DIR="${DROP_SWEEP_TMP}/drop${PCT}"
    echo "[INFO] Running drop_frac=${FRAC} (drop ${PCT}%)"
    conda run -n analysis python analysis/lib/predict_choice_from_item_prop_time_interactions.py \
      --root "${OUT_ROOT}" \
      --out-dir "${TMP_DIR}" \
      --value-source true \
      --feature-set location_interactions \
      --visit-type all \
      --n-bootstrap 0 \
      --n-sims 0 \
      --drop-frac "${FRAC}"
  done
  # Consolidate sweep results into a single CSV
  python3 -c "
import csv, os

pred_summary = '${PRED_SUMMARY_CSV}'
sweep_tmp = '${DROP_SWEEP_TMP}'
out_csv = '${DROP_SWEEP_CSV}'

rows = []

# 0% drop (full fixations) from the existing summary
with open(pred_summary) as f:
    reader = csv.DictReader(f)
    for r in reader:
        if r.get('feature_set', 'location_interactions') == 'location_interactions' and r.get('visit_type', 'all') == 'all':
            rows.append({'drop_pct': 0, 'retained_frac': 1.0, 'cv_mean': float(r['cv_mean']), 'cv_sem': float(r['cv_sem'])})
            break

# Each drop fraction
for pct in range(10, 100, 10):
    summary_path = os.path.join(sweep_tmp, f'drop{pct}', 'summary_prop_time_location_interactions_true_all_norm-within.csv')
    if not os.path.isfile(summary_path):
        print(f'Warning: missing summary for drop{pct}')
        continue
    with open(summary_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get('feature_set', 'location_interactions') == 'location_interactions' and r.get('visit_type', 'all') == 'all':
                rows.append({'drop_pct': pct, 'retained_frac': 1.0 - pct / 100.0, 'cv_mean': float(r['cv_mean']), 'cv_sem': float(r['cv_sem'])})
                break

rows.sort(key=lambda r: r['retained_frac'])

with open(out_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['drop_pct', 'retained_frac', 'cv_mean', 'cv_sem'])
    writer.writeheader()
    writer.writerows(rows)

print(f'Wrote {len(rows)} rows to {out_csv}')
"
  rm -rf "${DROP_SWEEP_TMP}"
fi

echo "[STEP] NN overview (behavior + eyegaze)"
OVERVIEW_ARGS=(
  --root "${OUT_ROOT}"
  --out-dir "${OVERVIEW_DIR}"
  --tag "${TAG}"
  --metric "${METRIC}"
  --human-data-dir "${REPO_ROOT}/output"
  --human-recalled-valence
  --drop-fix-pred-dir "${NN_DROPRECALL_DIR}"
)
conda run -n analysis python metarnn/lib/plot_NN_overview.py "${OVERVIEW_ARGS[@]}"

echo "[STEP] propDropSupplement figure"
if [[ -f "${DROP_SWEEP_CSV}" ]]; then
  conda run -n analysis python metarnn/lib/plot_prop_drop_supplement.py \
    --sweep-csv "${DROP_SWEEP_CSV}" \
    --human-pred-csv "${HUMAN_PRED_SUMMARY}" \
    --recall-csv "${RECALL_SIG_CSV}" \
    --out-dir "${OVERVIEW_DIR}" \
    --tag "${TAG}"
else
  echo "[WARN] Drop sweep CSV not found; skipping propDropSupplement."
fi

echo "[STEP] Humans vs NN comparison figure"
conda run -n analysis python metarnn/lib/plot_NN_H_comparison.py \
  --nn-root "${OUT_ROOT}" \
  --out-dir "${HCOMP_DIR}" \
  --tag "${TAG}"

echo "[DONE] NN figure generation complete."