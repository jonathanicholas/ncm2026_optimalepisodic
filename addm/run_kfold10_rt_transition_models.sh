#!/usr/bin/env bash
set -euo pipefail

# 10-fold cross-validation for the rt+transition model suite.
# Uses trial-level K-fold CV via addm/lib/run_kfold_cv.py.
#
# Usage:
#   conda activate analysis && ./addm/run_kfold10_rt_transition_models.sh
#
# Environment variables:
#   OUT_SUBDIR        - output subdirectory name
#   REWARD_SOURCE     - 'true' or 'recalled' (default: recalled)
#   SEED_START/N_RUNS - seed sweep parameters (default: 10 runs, seeds 123..132)
#   SEEDS             - explicit seed list (overrides SEED_START/N_RUNS)
#   SELECTION_SETTING - setting for best-seed selection (default: free3-rtTrans)

COMMON=(
  --mode fit
  --output-dir "output"
  --out-subdir "${OUT_SUBDIR:-}"
  --fixation-data-dir "${FIXATION_DATA_DIR:-data}"
  --n-folds 10
  --split-by subject
  --n-jobs "${N_JOBS:-10}"
  --n-sim-per-vbin 1000
  --theta-min 0.01
  --theta-max 0.99
  --time-col "${TIME_COL:-rt_ms}"
  --reward-source recalled
  --noise-param mu
  --resume
  --d-min "${D_MIN:-1e-4}"
  --d-max "${D_MAX:-0.05}"
  --sigma-min "${SIGMA_MIN:-1e-5}"
  --sigma-max "${SIGMA_MAX:-0.1}"
)

# Add --include-transition if INCLUDE_TRANSITION is set to a non-empty value.
if [[ -n "${INCLUDE_TRANSITION:-}" ]]; then
  COMMON+=(--include-transition)
fi

if [[ -n "${REWARD_SOURCE:-}" ]]; then
  COMMON+=(--reward-source "${REWARD_SOURCE}")
fi
if [[ -n "${DATA_DIR:-}" ]]; then
  COMMON+=(--data-dir "${DATA_DIR}")
fi

if [[ -n "${MU_MIN:-}" ]]; then
  COMMON+=(--mu-min "${MU_MIN}")
fi
if [[ -n "${MU_MAX:-}" ]]; then
  COMMON+=(--mu-max "${MU_MAX}")
fi

run_one () {
  local label="$1"; shift
  echo ""
  echo "============================================================"
  echo "[START] $(date '+%Y-%m-%d %H:%M:%S') | ${label}"
  echo "============================================================"
  python -m addm.lib.run_kfold_cv "${COMMON[@]}" "$@"
  echo "[DONE ] $(date '+%Y-%m-%d %H:%M:%S') | ${label}"
}

score_seed () {
  local summary_csv="$1"
  local expected_folds="$2"
  python - "$summary_csv" "$expected_folds" <<'PY'
import math
import sys
from pathlib import Path

import pandas as pd

csv_path = Path(sys.argv[1])
expected_folds = int(sys.argv[2])

if not csv_path.exists():
    print("-inf")
    sys.exit(0)

try:
    df = pd.read_csv(csv_path)
except Exception:
    print("-inf")
    sys.exit(0)

if "loglik_test" not in df.columns or "fold" not in df.columns:
    print("-inf")
    sys.exit(0)

df = df.copy()
df["fold"] = pd.to_numeric(df["fold"], errors="coerce")
df["loglik_test"] = pd.to_numeric(df["loglik_test"], errors="coerce")
df = df.dropna(subset=["fold", "loglik_test"])

# If the summary was written incrementally (e.g., with --resume),
# be robust to any accidental duplicate fold rows.
df = df.sort_values("fold").drop_duplicates(subset=["fold"], keep="last")

n_folds_done = df["fold"].nunique()
if n_folds_done < expected_folds:
    # Treat incomplete runs as invalid for selection.
    print("-inf")
    sys.exit(0)

mean_ll = float(df["loglik_test"].mean())
if not math.isfinite(mean_ll):
    print("-inf")
else:
    print(f"{mean_ll:.12f}")
PY
}

copy_best_outputs () {
  local best_run_dir="$1"
  local base_out_dir="$2"
  mkdir -p "$base_out_dir"

  # Copy summary + fold assignment CSVs back to the base output folder,
  # preserving filenames so downstream scripts keep working unchanged.
  shopt -s nullglob
  local csvs=("$best_run_dir"/addm_kfold_*.csv)
  if (( ${#csvs[@]} == 0 )); then
    echo "[ERROR] No addm_kfold_*.csv files found in best run dir: $best_run_dir" >&2
    exit 1
  fi
  cp -f "${csvs[@]}" "$base_out_dir"/
  shopt -u nullglob
}

BASE_OUT_SUBDIR="${OUT_SUBDIR:-}"
N_RUNS="${N_RUNS:-10}"
SEED_START="${SEED_START:-123}"
SELECTION_SETTING="${SELECTION_SETTING:-free3-rtTrans}"
SPLIT_SEED="${SPLIT_SEED:-123}"
EXPECTED_FOLDS=10

declare -a SEED_LIST
if [[ -n "${SEEDS:-}" ]]; then
  # Allow either space- or comma-separated.
  read -r -a SEED_LIST <<< "${SEEDS//,/ }"
else
  for ((i=0; i<"$N_RUNS"; i++)); do
    SEED_LIST+=("$((SEED_START + i))")
  done
fi

if (( ${#SEED_LIST[@]} == 0 )); then
  echo "[ERROR] No seeds specified (SEEDS empty and N_RUNS=0?)" >&2
  exit 1
fi

BASE_OUT_DIR="output/addm/kfold"
if [[ -n "${BASE_OUT_SUBDIR}" ]]; then
  BASE_OUT_DIR+="/${BASE_OUT_SUBDIR}"
fi
RUNS_DIR="${BASE_OUT_DIR}/runs"
mkdir -p "$RUNS_DIR"

SEED_SCORES_CSV="${RUNS_DIR}/seed_scores_splitSeed${SPLIT_SEED}.csv"
echo "seed,mean_loglik_test_free3,mean_loglik_test_theta1,summary_free3,summary_theta1" > "$SEED_SCORES_CSV"

best_seed=""
best_score="-inf"

for seed in "${SEED_LIST[@]}"; do
  run_subdir=""
  if [[ -n "${BASE_OUT_SUBDIR}" ]]; then
    run_subdir="${BASE_OUT_SUBDIR}/runs/seed${seed}"
  else
    run_subdir="runs/seed${seed}"
  fi
  run_dir="output/addm/kfold/${run_subdir}"

  echo ""
  echo "############################################################"
  echo "[RUN ] seed=${seed} | out-subdir=${run_subdir}"
  echo "############################################################"

  # 1) theta fixed at 1 (baseline irrelevant mode)
  run_one "theta1 rt+transition (kfold10, seed=${seed})" \
    --out-subdir "${run_subdir}" \
    --seed "${seed}" \
    --split-seed "${SPLIT_SEED}" \
    --fix-theta 1 \
    --theta-max 1.0 \
    --irrelevant-mode zero \
    --setting theta1-rtTrans

  # 2) free3 baseline (irrelevant drift=0)
  run_one "free3 rt+transition (kfold10, seed=${seed})" \
    --out-subdir "${run_subdir}" \
    --seed "${seed}" \
    --split-seed "${SPLIT_SEED}" \
    --irrelevant-mode zero \
    --setting free3-rtTrans

  summary_free3="${run_dir}/addm_kfold_fit_summary_free3-rtTrans.csv"
  summary_theta1="${run_dir}/addm_kfold_fit_summary_theta1-rtTrans.csv"
  mean_ll_free3=$(score_seed "$summary_free3" "$EXPECTED_FOLDS")
  mean_ll_theta1=$(score_seed "$summary_theta1" "$EXPECTED_FOLDS")

  echo "${seed},${mean_ll_free3},${mean_ll_theta1},${summary_free3},${summary_theta1}" >> "$SEED_SCORES_CSV"
  echo "[SCORE] seed=${seed} | mean_ll_free3=${mean_ll_free3} | mean_ll_theta1=${mean_ll_theta1} | split_seed=${SPLIT_SEED}"

  # Choose which model's score determines the "best" seed to copy to the top-level.
  summary_csv="${run_dir}/addm_kfold_fit_summary_${SELECTION_SETTING}.csv"
  mean_ll=$(score_seed "$summary_csv" "$EXPECTED_FOLDS")

  # Track best (higher loglik is better).
  best_pair=$(python - "$best_seed" "$best_score" "$seed" "$mean_ll" <<'PY'
import math
import sys

best_seed = sys.argv[1]
best_score = float(sys.argv[2]) if sys.argv[2] != "-inf" else float("-inf")
seed = sys.argv[3]
score = float(sys.argv[4]) if sys.argv[4] != "-inf" else float("-inf")

if math.isfinite(score) and (not math.isfinite(best_score) or score > best_score):
    print(seed, f"{score:.12f}")
else:
    print(best_seed, f"{best_score:.12f}" if math.isfinite(best_score) else "-inf")
PY
  )
  read -r best_seed best_score <<< "$best_pair"
done

if [[ -z "$best_seed" || "$best_score" == "-inf" ]]; then
  echo "[ERROR] Could not identify a best seed (all runs incomplete or missing summary CSVs)." >&2
  echo "        See: $SEED_SCORES_CSV" >&2
  exit 1
fi

best_run_dir="${RUNS_DIR}/seed${best_seed}"
echo ""
echo "============================================================"
echo "[BEST] seed=${best_seed} | mean_loglik_test(${SELECTION_SETTING})=${best_score}"
echo "       best_run_dir=${best_run_dir}"
echo "============================================================"

copy_best_outputs "$best_run_dir" "$BASE_OUT_DIR"
echo "seed=${best_seed}" > "${BASE_OUT_DIR}/best_seed.txt"
echo "split_seed=${SPLIT_SEED}" >> "${BASE_OUT_DIR}/best_seed.txt"
echo "mean_loglik_test_${SELECTION_SETTING}=${best_score}" >> "${BASE_OUT_DIR}/best_seed.txt"
echo "seed_scores_csv=${SEED_SCORES_CSV}" >> "${BASE_OUT_DIR}/best_seed.txt"

# # Optional additional variants (match the LOGO script comments):
# run_one "free3 irrelSumrel rt+transition (kfold10)" \
#   --irrelevant-mode sumrel \
#   --setting free3-irrelSumrel-rtTrans

# run_one "free3 irrelThetaSumrel rt+transition (kfold10)" \
#   --irrelevant-mode theta_sumrel \
#   --setting free3-irrelThetaSumrel-rtTrans

echo ""
echo "[OK] All kfold10 rtTrans runs finished. CSVs should be in output/addm/kfold/."
