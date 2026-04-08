#!/bin/bash

# Resolve paths relative to this script's location
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Use environment variables if set, otherwise defaults relative to project root
file_path="${FILE_PATH:-$BASE_DIR/output}"
task_path="${TASK_PATH:-$BASE_DIR/task/emdm-eyetracking/game_info}"
data_path="${DATA_PATH:-$BASE_DIR/data}"
img_path="${IMG_PATH:-}"
output_base_dir="${OUTPUT_DIR:-$BASE_DIR/output}"
roi_type="${ROI_TYPE:-original}"
buffer_size="${BUFFER_SIZE:-50}"
max_parallel="${MAX_PARALLEL:-4}"

# Build subjects array from exported SUBJECTS (space-separated string)
if [ -n "$SUBJECTS" ]; then
    read -r -a subjects <<< "$SUBJECTS"
else
    subjects=()
fi

# Debug: show raw SUBJECTS env and parsed count
echo "[run_analyze_recall] Raw SUBJECTS env: '$SUBJECTS'"
echo "[run_analyze_recall] Parsed subjects count: ${#subjects[@]}"

# Fallback: if no subjects parsed, attempt to auto-detect subject directories in data_path
if [ ${#subjects[@]} -eq 0 ]; then
    echo "[run_analyze_recall] SUBJECTS empty. Attempting auto-detect in $data_path ..."
    if [ -d "$data_path" ]; then
        while IFS= read -r d; do
            subj="$(basename "$d")"
            if [[ $subj =~ ^[0-9]{3}$ ]]; then
                subjects+=("$subj")
            fi
        done < <(find "$data_path" -maxdepth 1 -mindepth 1 -type d 2>/dev/null)
    fi
    echo "[run_analyze_recall] Auto-detected subjects: ${subjects[*]} (count=${#subjects[@]})"
fi

# Guard: if still empty, warn and exit
if [ ${#subjects[@]} -eq 0 ]; then
    echo "[run_analyze_recall] ERROR: No subjects provided or detected. Exiting." >&2
    exit 1
fi

# Function to process a single subject's eyetracking data

analyze_recall() {
    local subj=$1
    echo "Starting recall analysis for subject $subj..."
    if [ "$MAKE_RECALL_ANIMATIONS" = true ]; then
        python "$SCRIPT_DIR/lib/analyze_recall.py" --participant "$subj" --file_path "$file_path" --task_path "$task_path" --data_path "$data_path" --img_path "$img_path" --make_recall_animations
    else
        python "$SCRIPT_DIR/lib/analyze_recall.py" --participant "$subj" --file_path "$file_path" --task_path "$task_path" --data_path "$data_path" --img_path "$img_path"
    fi
    echo "Finished recall analysis for subject $subj"
}

# Process subjects in parallel

echo "Starting individual subject analyses..."
for subj in "${subjects[@]}"; do
    # Count current running processes
    while [ $(jobs -r | wc -l) -ge $max_parallel ]; do
        sleep 1
    done
    analyze_recall "$subj" &
done

# Wait for all individual analyses to complete
wait
echo "All individual recall analyses complete!"

# Run group analysis
echo ""
echo "Starting group analysis..."
python "$SCRIPT_DIR/lib/analyze_recall_group.py" --output_base_dir "$output_base_dir" --data_dir "$data_path" --roi_type "$roi_type" --buffer_size "$buffer_size"
echo "Group analysis complete!"

echo ""
echo "All analyses complete!"
