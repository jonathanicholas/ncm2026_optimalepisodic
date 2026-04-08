"""Generate wedge-aligned fixation CSVs for all subjects.

Reads fixation data and free recall data from data/SUBID/ and writes
wedge-aligned fixation CSVs back to data/SUBID/.

Usage (from the repository root):
    python analysis/lib/generate_wedge_aligned_fixations.py

Or for a single subject:
    python analysis/lib/generate_wedge_aligned_fixations.py --participant 101
"""

import os
import argparse
import json

import numpy as np
import pandas as pd


def generate_circle_positions(radius: int = 380, n_positions: int = 6):
    """Generate n evenly spaced positions around a circle.

    Starting from top (90 degrees) and going clockwise.
    """
    positions = []
    angles = []
    angle_order = [90, 30, -30, -90, -150, 150]

    for angle_deg in angle_order:
        angle = np.deg2rad(angle_deg)
        x = int(radius * np.cos(angle))
        y = int(radius * np.sin(angle))
        positions.append((x, y))
        angles.append(angle_deg)

    return positions, angles


def load_fixation_dataframe(subj: str, data_dir: str, buffer_size: int, roi_type: str) -> pd.DataFrame:
    """Load the fixation dataframe from data/SUBID/."""
    subj_dir = os.path.join(data_dir, subj)
    if roi_type == "equal_area":
        filename = os.path.join(subj_dir, f"{subj}_fixations_df_equal_area_buffer_{buffer_size}.csv")
    elif roi_type == "original":
        filename = os.path.join(subj_dir, f"{subj}_fixations_df_original_buffer_{buffer_size}.csv")
    else:
        raise ValueError(f"Invalid roi_type: {roi_type}. Must be 'equal_area' or 'original'")

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Fixation dataframe not found: {filename}")

    return pd.read_csv(filename)


def collect_rotated_fixations(
    subj: str,
    data_dir: str,
    task_path: str,
    buffer_size: int,
    roi_type: str,
    pre_recall_window: int,
    post_recall_window: int,
    bin_size: int,
):
    """Collect fixation locations, rotated so that the recalled item's wedge is canonical.

    Returns a DataFrame with one row per fixation used for the heatmaps, containing:
        - x_rot, y_rot : rotated coordinates in canonical wedge frame (center at 0,0; y up)
        - time_rel     : time relative to recall onset (ms)
        - bin_center   : 100 ms bin center (or user-specified bin_size)
        - duration     : fixation duration (ms), taken from fix_duration_bounded
    Only fixations on item locations (roi_content not in {none, fixation, NaN}) are included.
    """
    fixations_df = load_fixation_dataframe(subj, data_dir, buffer_size, roi_type)
    free_recall_fixations = fixations_df[fixations_df["event"] == "free_recall"].copy()

    if free_recall_fixations.empty:
        print(f"Warning: No free recall fixations found for subject {subj}")
        return pd.DataFrame()

    free_recall_fixations["fix_onset_relative"] = (
        free_recall_fixations["fix_start"] - free_recall_fixations["eyetracker_onset"]
    )

    # Free recall CSV is at data/SUBID/SUBID_freerecall.csv in this directory layout
    free_recall_csv = os.path.join(data_dir, subj, f"{subj}_freerecall.csv")
    free_recall_df = pd.read_csv(free_recall_csv)
    free_recall_df["onset"] = (free_recall_df["onset"] * 1000).astype(int)
    free_recall_df["offset"] = (free_recall_df["offset"] * 1000).astype(int)

    # Load game/image information
    with open(os.path.join(task_path, f"emdm-eyetracking/game_info/games_{subj}.json"), "r") as f:
        images_by_game = json.load(f)
    with open(os.path.join(task_path, f"emdm-eyetracking/game_info/positions_{subj}.json"), "r") as f:
        image_positions = json.load(f)

    # Screen / ROI geometry
    center_x = 3840 / 2.0
    center_y = 2160 / 2.0
    circle_radius = 380
    positions, angles = generate_circle_positions(radius=circle_radius, n_positions=6)

    time_min = -pre_recall_window
    time_max = post_recall_window
    bins = np.arange(time_min, time_max + bin_size, bin_size)

    all_rows = []

    game_numbers = sorted(free_recall_df["game"].unique())

    for curr_game in game_numbers:
        curr_images = images_by_game[curr_game - 1]
        curr_recalls = free_recall_df[free_recall_df["game"] == curr_game]
        game_fixations = free_recall_fixations[free_recall_fixations["game"] == curr_game]

        if curr_recalls.empty or game_fixations.empty:
            continue

        sorted_recalls = curr_recalls.sort_values("onset")

        for _, recall in sorted_recalls.iterrows():
            onset = recall["onset"]
            offset = recall["offset"]
            recalled_item = recall["item"]

            if recalled_item not in curr_images:
                continue

            extended_mask = (
                (game_fixations["fix_onset_relative"] >= onset - pre_recall_window)
                & (game_fixations["fix_onset_relative"] <= min(onset + post_recall_window, offset))
            )

            for _, other_recall in sorted_recalls.iterrows():
                other_onset = other_recall["onset"]
                other_offset = other_recall["offset"]
                if other_onset == onset:
                    continue
                overlap_in_pre = (
                    (game_fixations["fix_onset_relative"] >= onset - pre_recall_window)
                    & (game_fixations["fix_onset_relative"] < onset)
                    & (game_fixations["fix_onset_relative"] >= other_onset)
                    & (game_fixations["fix_onset_relative"] <= other_offset)
                )
                extended_mask = extended_mask & (~overlap_in_pre)

            window_fixations = game_fixations[extended_mask].copy()
            if window_fixations.empty:
                continue

            window_fixations["time_rel"] = window_fixations["fix_onset_relative"] - onset

            valid_mask = (
                window_fixations["roi_content"].notna()
                & (window_fixations["roi_content"] != "none")
                & (window_fixations["roi_content"] != "fixation")
            )
            window_fixations = window_fixations[valid_mask]
            if window_fixations.empty:
                continue

            target_pos_index = image_positions[recalled_item]
            target_angle = angles[target_pos_index]
            canonical_angle = angles[0]  # top position (90 degrees)
            rotation_deg = canonical_angle - target_angle
            rotation_rad = np.deg2rad(rotation_deg)

            for _, fix in window_fixations.iterrows():
                time_rel = fix["time_rel"]
                if time_rel < time_min or time_rel > time_max:
                    continue

                bin_idx = np.digitize(time_rel, bins) - 1
                if bin_idx < 0 or bin_idx >= len(bins) - 1:
                    continue
                bin_center = (bins[bin_idx] + bins[bin_idx + 1]) / 2.0

                fx = fix["fix_x"] - center_x
                fy = center_y - fix["fix_y"]

                x_rot = fx * np.cos(rotation_rad) - fy * np.sin(rotation_rad)
                y_rot = fx * np.sin(rotation_rad) + fy * np.cos(rotation_rad)

                all_rows.append(
                    {
                        "subject": subj,
                        "game": curr_game,
                        "recalled_item": recalled_item,
                        "time_rel": float(time_rel),
                        "bin_center": float(bin_center),
                        "x_rot": float(x_rot),
                        "y_rot": float(y_rot),
                        "duration": float(fix["fix_duration_bounded"]),
                    }
                )

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


def find_subjects(data_dir: str) -> list[str]:
    """Return sorted list of 3-digit subject IDs that have fixation data."""
    subjects = []
    for entry in os.listdir(data_dir):
        if entry.isdigit() and len(entry) == 3:
            subj_dir = os.path.join(data_dir, entry)
            if os.path.isdir(subj_dir):
                subjects.append(entry)
    return sorted(subjects)


def main():
    parser = argparse.ArgumentParser(
        description="Generate wedge-aligned fixation CSVs for all (or one) subject(s)."
    )
    parser.add_argument("--participant", type=str, default=None,
                        help="Single participant ID (e.g., 101). Omit to process all.")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Path to data directory containing per-subject folders")
    parser.add_argument("--task_path", type=str, default="task",
                        help="Path to task/game info directory")
    parser.add_argument("--buffer_size", type=int, default=50)
    parser.add_argument("--roi_type", type=str, default="original",
                        choices=["original", "equal_area"])
    parser.add_argument("--pre_recall_window", type=int, default=3000)
    parser.add_argument("--post_recall_window", type=int, default=750)
    parser.add_argument("--bin_size", type=int, default=100)

    args = parser.parse_args()

    if args.participant:
        subjects = [args.participant]
    else:
        subjects = find_subjects(args.data_dir)

    print(f"Processing {len(subjects)} subject(s)")

    for subj in subjects:
        print(f"  {subj}...", end=" ", flush=True)
        try:
            rotated_df = collect_rotated_fixations(
                subj=subj,
                data_dir=args.data_dir,
                task_path=args.task_path,
                buffer_size=args.buffer_size,
                roi_type=args.roi_type,
                pre_recall_window=args.pre_recall_window,
                post_recall_window=args.post_recall_window,
                bin_size=args.bin_size,
            )
        except FileNotFoundError as e:
            print(f"SKIP ({e})")
            continue

        if rotated_df.empty:
            print("no data")
            continue

        out_csv = os.path.join(
            args.data_dir, subj,
            f"{subj}_wedge_aligned_fixations_{args.roi_type}_buffer_{args.buffer_size}.csv",
        )
        rotated_df.to_csv(out_csv, index=False)
        print(f"OK ({len(rotated_df)} rows)")

    print("Done.")


if __name__ == "__main__":
    main()
