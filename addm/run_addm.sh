#!/usr/bin/env bash
# Master script for running aDDM analyses.
# All commands assume they are run from the repository root directory.
#
# Usage:
#   bash addm/run_addm.sh fix        # fixation time (--time-col fix_ms)
#   bash addm/run_addm.sh rtTrans    # RT with transitions (--time-col rt_ms --include-transition)
set -euo pipefail

cd "$(dirname "$0")/.."  # cd to repo root

# ── Parse mode ───────────────────────────────────────────────────────────────
MODE="${1:?Usage: $0 <fix|rtTrans>}"
case "$MODE" in
  fix)
    TIME_COL="fix_ms"
    TRANSITION_FLAG=""
    INCLUDE_TRANSITION=""
    OUT_SUBDIR="fix_recalled_final"
    ;;
  rtTrans)
    TIME_COL="rt_ms"
    TRANSITION_FLAG="--include-transition"
    INCLUDE_TRANSITION="1"
    OUT_SUBDIR="rtTrans_recalled_final"
    ;;
  *)
    echo "Error: mode must be 'fix' or 'rtTrans', got '$MODE'" >&2
    exit 1
    ;;
esac

# ── 1. K-fold cross-validation ───────────────────────────────────────────────
# 10-fold CV with seed sweep (5 seeds), best seed selected by mean held-out LL.
# Output: output/addm/kfold/${OUT_SUBDIR}/
OUT_SUBDIR="$OUT_SUBDIR" \
TIME_COL="$TIME_COL" \
INCLUDE_TRANSITION="$INCLUDE_TRANSITION" \
SPLIT_SEED=123 \
SEED_START=1000 \
N_RUNS=5 \
D_MIN=1e-5 \
D_MAX=1e-3 \
MU_MIN=1 \
MU_MAX=100 \
N_JOBS=10 \
  bash addm/run_kfold10_rt_transition_models.sh

# ── 2. Compare CV fits ──────────────────────────────────────────────────────
# Merge per-fold held-out log-likelihoods into wide and summary tables.
# Output: output/addm/kfold_compare/${OUT_SUBDIR}/
python -m addm.compare_cv_fits \
  --cv-dir "output/addm/kfold/${OUT_SUBDIR}" \
  --out-dir "output/addm/kfold_compare/${OUT_SUBDIR}" \
  --mode fit

# Bayesian comparison of aDDM vs DDM
Rscript addm/run_mixed_effects_addm_comparison.R \
  --wide-csv "output/addm/kfold_compare/${OUT_SUBDIR}/cv_compare_by_game_wide.csv" \
  --out-dir "output/addm/kfold_compare/${OUT_SUBDIR}"

# Plot kfold parameter bars (free3 model: theta, d, sigma mean ± SEM)
python -m addm.plot_kfold_parameter_bars \
  --kfold-summary-csv "output/addm/kfold/${OUT_SUBDIR}/addm_kfold_fit_summary_free3-rtTrans.csv" \
  --out-dir "output/addm/kfold_compare/${OUT_SUBDIR}" \
  --tag "${OUT_SUBDIR}_free3"

# ── 3. Posterior predictive comparison (aDDM vs DDM) ─────────────────────────
# Simulates from mean kfold parameters, plots choice + RT psychometrics + ΔLL.
# Output: output/addm/ppc/
python -m addm.plot_addm_ddm_comparison \
  --addm-rt-kfold-dir "output/addm/kfold/${OUT_SUBDIR}" \
  --kfold-compare-wide-csv "output/addm/kfold_compare/${OUT_SUBDIR}/cv_compare_by_game_wide.csv" \
  --out-dir "output/addm/ppc" \
  --tag "${OUT_SUBDIR}"

# ── 4. Parameter recovery sweep ──────────────────────────────────────────────
# 500 combos, 1 rep each, mu parameterization, recalled rewards.
# Output: output/addm/parameter_recovery_sweep/${OUT_SUBDIR}/
python -m addm.lib.parameter_recovery_sweep \
  --output-dir output \
  --fixation-data-dir data \
  --run-tag "$OUT_SUBDIR" \
  --time-col "$TIME_COL" \
  $TRANSITION_FLAG \
  --units ms \
  --n-combos 1000 \
  --reps-per-combo 1 \
  --seed 123 \
  --noise-param mu \
  --d-min 1e-5 \
  --d-max 1e-3 \
  --theta-min 0.01 \
  --theta-max 0.99 \
  --mu-min 1 \
  --mu-max 100 \
  --fit-d-min 1e-5 \
  --fit-d-max 1e-3 \
  --fit-theta-min 0.01 \
  --fit-theta-max 0.99 \
  --fit-sigma-min 1e-7 \
  --fit-sigma-max 10 \
  --fit-mu-min 1 \
  --fit-mu-max 100 \
  --d0 5e-4 \
  --theta0 0.5 \
  --mu0 50 \
  --n-v-bins 7 \
  --rt-bins-max 15 \
  --rt-bins-fixed 0 \
  --min-trials-per-rt-bin 25 \
  --n-sim-per-vbin 1000 \
  --alpha 1.0 \
  --n-jobs 10 \
  --reward-source recalled \
  --resume

# Plot parameter recovery results
python -m addm.plot_recovery_sweep \
  --sweep-csv "output/addm/parameter_recovery_sweep/${OUT_SUBDIR}/sweep_runs.csv" \
  --aggregate mean \
  --format pdf

# ── 5. Supplement figure (rtTrans only) ───────────────────────────────────────
# Combines PPC, fit params (rtTrans), and parameter recovery (rtTrans) into a
# single supplementary figure.
if [[ "$MODE" == "rtTrans" ]]; then
  python -m addm.plot_addm_supplement \
    --out-dir "output/addm"
fi
