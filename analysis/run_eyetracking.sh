#!/bin/bash
# Run the full eyetracking analysis pipeline and generate Figure2.pdf.
#
# Pipeline:
#   1. analyze_recall.py (per subject)     — recall fixation analysis
#   2. analyze_recall_group.py             — group time course + cluster test
#   3. generate_wedge_aligned_fixations.py — wedge-rotated fixations for heatmaps
#   4. prepare_choice_fixations.py         — aggregate choice fixations
#   5. choice_fixation_proportions.py      — fixation proportions by relevance/valence
#   6. predict_choice_from_item_prop_time_interactions.py — CV logistic regression
#   7. run_mixed_effects_eye.R             — Bayesian mixed-effects models (brms)
#   8. analyze_eyetracking.py              — generate Figure2.pdf
#
# Outputs:
#   output/eyegaze/Figure2.pdf
#   output/eyegaze/stats/*.csv  (model summaries + coefficients)
#   output/eyegaze/recall/      (group time course + cluster stats)
#
# Usage:
#   conda activate analysis
#   bash analysis/run_eyetracking.sh
#
# Options (environment variables):
#   MAX_PARALLEL=4       max parallel subject jobs (default: 4)
#   N_BOOTSTRAP=1000     bootstrap replicates for CV model CIs (default: 1000)
#   SKIP_SUBJECTS=false  skip per-subject steps if outputs exist (default: false)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$BASE_DIR"

MAX_PARALLEL="${MAX_PARALLEL:-4}"
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"
SKIP_SUBJECTS="${SKIP_SUBJECTS:-false}"
ROI_TYPE="original"
BUFFER_SIZE=50

# Discover subjects
subjects=()
for d in data/[0-9][0-9][0-9]; do
    [ -d "$d" ] && subjects+=("$(basename "$d")")
done

if [ ${#subjects[@]} -eq 0 ]; then
    echo "ERROR: No subject directories found in data/" >&2
    exit 1
fi
echo "Found ${#subjects[@]} subjects"

# Subjects excluded from eyetracking analyses
EXCLUDE="107 131"

# ─────────────────────────────────────────────────────
# Step 1: Per-subject recall analysis
# ─────────────────────────────────────────────────────
echo ""
echo "=== Step 1/${#subjects[@]}: Per-subject recall analysis ==="

for subj in "${subjects[@]}"; do
    if [[ " $EXCLUDE " == *" $subj "* ]]; then
        continue
    fi
    # Skip if outputs exist
    if [ "$SKIP_SUBJECTS" = "true" ] && \
       [ -f "output/${subj}/${subj}_fixation_time_course_${ROI_TYPE}_buffer_${BUFFER_SIZE}.csv" ]; then
        echo "[SKIP] analyze_recall for $subj (output exists)"
        continue
    fi
    while [ "$(jobs -r | wc -l)" -ge "$MAX_PARALLEL" ]; do
        sleep 1
    done
    python analysis/lib/analyze_recall.py \
        --participant "$subj" \
        --file_path output \
        --task_path "task/emdm-eyetracking/game_info" \
        --data_path data \
        --img_path "task/images" \
        --buffer_size "$BUFFER_SIZE" \
        --roi_type "$ROI_TYPE" &
done
wait
echo "[DONE] Per-subject recall analysis"

# ─────────────────────────────────────────────────────
# Step 2: Group recall analysis (time course + cluster test)
# ─────────────────────────────────────────────────────
echo ""
echo "=== Step 2: Group recall analysis ==="
python analysis/lib/analyze_recall_group.py \
    --output_base_dir output \
    --data_dir data \
    --roi_type "$ROI_TYPE" \
    --buffer_size "$BUFFER_SIZE"
echo "[DONE] analyze_recall_group.py"

# ─────────────────────────────────────────────────────
# Step 3: Generate wedge-aligned fixations
# ─────────────────────────────────────────────────────
echo ""
echo "=== Step 3: Wedge-aligned fixations ==="
python analysis/lib/generate_wedge_aligned_fixations.py \
    --data_dir data \
    --task_path task \
    --buffer_size "$BUFFER_SIZE" \
    --roi_type "$ROI_TYPE"
echo "[DONE] generate_wedge_aligned_fixations.py"

# ─────────────────────────────────────────────────────
# Step 4: Prepare choice fixations
# ─────────────────────────────────────────────────────
echo ""
echo "=== Step 4: Prepare choice fixations ==="
python analysis/lib/prepare_choice_fixations.py \
    --base-dir .
echo "[DONE] prepare_choice_fixations.py"

# ─────────────────────────────────────────────────────
# Step 5: Choice fixation proportions (uses recalled valence)
# ─────────────────────────────────────────────────────
echo ""
echo "=== Step 5: Choice fixation proportions ==="
python analysis/lib/choice_fixation_proportions.py \
    --metric duration
echo "[DONE] choice_fixation_proportions.py"

# ─────────────────────────────────────────────────────
# Step 6: CV logistic regression (prop-time x reward x relevance)
# ─────────────────────────────────────────────────────
echo ""
echo "=== Step 6: Choice prediction from prop-time interactions ==="
python analysis/lib/predict_choice_from_item_prop_time_interactions.py \
    --out-dir output/eyegaze/stats \
    --value-source recalled \
    --feature-set location_interactions \
    --visit-type all \
    --n-bootstrap "$N_BOOTSTRAP" \
    --n-sims 0
echo "[DONE] predict_choice_from_item_prop_time_interactions.py"

# ─────────────────────────────────────────────────────
# Step 7: Generate Figure2.pdf
# ─────────────────────────────────────────────────────
echo ""
echo "=== Step 7: Generate Figure2.pdf ==="
python analysis/analyze_eyetracking.py \
    --buffer-ms "$BUFFER_SIZE" \
    --out-dir output/eyegaze
echo "[DONE] analyze_eyetracking.py"

# ─────────────────────────────────────────────────────
# Step 8: Mixed-effects models (brms)
# ─────────────────────────────────────────────────────
echo ""
echo "=== Step 8: Mixed-effects eye models ==="
Rscript analysis/run_mixed_effects_eye.R
echo "[DONE] run_mixed_effects_eye.R"

echo ""
echo "=== Eyetracking pipeline complete ==="
echo "  Figure:  output/eyegaze/Figure2.pdf"
echo "  Stats:   output/eyegaze/stats/"
echo "  Recall:  output/eyegaze/recall/"
