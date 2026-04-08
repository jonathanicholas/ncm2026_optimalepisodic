import os
import argparse
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import gaussian_filter
from recalled_valence import build_recalled_valence_map


def generate_circle_positions(radius: int = 380, n_positions: int = 6):
    """Generate n evenly spaced positions around a circle.

    Starting from top (90 degrees) and going clockwise.
    Returns
    -------
    positions : list of (x, y)
        Cartesian coordinates in a math-style system (x right, y up) relative to center.
    angles : list of float
        Angles in degrees for each position (math convention, 0 = right, 90 = up).
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


def draw_full_roi(ax, outer_radius: int, central_radius: int) -> None:
    """Draw full circular ROI (outer/central circles and 6 sector lines).

    Coordinates are in the rotated frame with center at (0, 0), y up.
    Matches the geometry used in the recall wedge visualizations.
    """
    outer = Circle((0, 0), outer_radius, fill=False, edgecolor="black", linewidth=2)
    ax.add_patch(outer)

    for i in range(6):
        angle_deg = i * 60 - 30
        rad = math.radians(90 - angle_deg)
        x_end = outer_radius * math.cos(rad)
        y_end = outer_radius * math.sin(rad)
        ax.plot([0, x_end], [0, y_end], color="black", linewidth=2)

    center = Circle((0, 0), central_radius, fill=True, edgecolor="black", facecolor="white", linewidth=2, zorder=100)
    ax.add_patch(center)


def _find_subjects_with_fixations(output_base_dir: str, data_dir: str | None = None) -> list[str]:
    """Return list of subject IDs that have fixation CSVs.

    If *data_dir* is provided, per-subject directories are searched there
    instead of *output_base_dir*.
    """
    search_dir = data_dir if data_dir is not None else output_base_dir
    subjects: list[str] = []
    for entry in os.listdir(search_dir):
        subj_dir = os.path.join(search_dir, entry)
        if not os.path.isdir(subj_dir):
            continue
        if entry.isdigit() and len(entry) == 3:
            subjects.append(entry)
    subjects.sort()
    return subjects


def _load_choice_fixation_dataframe(subj: str, output_base_dir: str, buffer_size: int, roi_type: str, data_dir: str | None = None) -> pd.DataFrame:
    """Load fixation dataframe for the choice phase.

    If *data_dir* is provided, per-subject data is loaded from there instead
    of *output_base_dir*.
    """
    search_dir = data_dir if data_dir is not None else output_base_dir
    subj_dir = os.path.join(search_dir, subj)
    if roi_type == "equal_area":
        if buffer_size == 0:
            filename = f"{subj_dir}/{subj}_fixations_df_equal_area_no_buffer.csv"
        else:
            filename = f"{subj_dir}/{subj}_fixations_df_equal_area_buffer_{buffer_size}.csv"
    elif roi_type == "original":
        if buffer_size == 0:
            filename = f"{subj_dir}/{subj}_fixations_df_original_no_buffer.csv"
        else:
            filename = f"{subj_dir}/{subj}_fixations_df_original_buffer_{buffer_size}.csv"
    else:
        raise ValueError(f"Invalid roi_type: {roi_type}. Must be 'equal_area' or 'original'")

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Fixation dataframe not found: {filename}")

    return pd.read_csv(filename)


def _get_total_item_fix_time(subj: str, output_base_dir: str, buffer_size: int, roi_type: str, data_dir: str | None = None) -> float:
    """Return total fixation duration on item locations during choice for a subject.

    This is used as a common normalization denominator so that contrast and
    interaction maps are expressed as proportions of *total* fixation time
    across all items, not just within a given subset.
    """

    try:
        fix_df = _load_choice_fixation_dataframe(subj, output_base_dir, buffer_size, roi_type, data_dir=data_dir)
    except FileNotFoundError:
        return 0.0

    choice_fix = fix_df[(fix_df["phase"] == "choice") & (fix_df["event"] == "choice")].copy()
    if choice_fix.empty:
        return 0.0

    valid = (
        choice_fix["roi_content"].notna()
        & (choice_fix["roi_content"] != "none")
        & (choice_fix["roi_content"] != "fixation")
    )
    image_fix = choice_fix[valid].copy()
    if image_fix.empty:
        return 0.0

    dur_col = "fix_duration_bounded" if "fix_duration_bounded" in image_fix.columns else "fix_duration_full"
    return float(image_fix[dur_col].sum())


def _load_game_and_position_info(subj: str, task_path: str):
    """Load game image lists and image->position index mapping for a subject."""
    game_info_dir = os.path.join(task_path, "emdm-eyetracking", "game_info")
    games_path = os.path.join(game_info_dir, f"games_{subj}.json")
    positions_path = os.path.join(game_info_dir, f"positions_{subj}.json")

    if not os.path.exists(games_path):
        raise FileNotFoundError(f"Missing games JSON for subject {subj}: {games_path}")
    if not os.path.exists(positions_path):
        raise FileNotFoundError(f"Missing positions JSON for subject {subj}: {positions_path}")

    with open(games_path, "r") as f:
        games = json.load(f)
    with open(positions_path, "r") as f:
        image_positions = json.load(f)

    return games, image_positions

def _build_valence_map_for_choice(subj: str, root: str) -> dict[tuple, str]:
    """Return {(game, image): valence_str} for a subject.

    Uses recalled values with true outcome as fallback.
    """
    data_dir = os.path.join(root, "data")
    recalled_map = build_recalled_valence_map(subj, data_dir)
    # Convert (valence_str, value) tuples to just valence_str
    return {k: v[0] for k, v in recalled_map.items()}


def collect_choice_rotated_fixations(
    subj: str,
    output_base_dir: str,
    task_path: str,
    buffer_size: int,
    roi_type: str,
    anchor_type: str,
    anchor_valence_filter: tuple[str, ...] | None = None,
    data_dir: str | None = None,
) -> pd.DataFrame:
    """Collect wedge-aligned choice fixations for one subject.

    For each choice trial, we:
      - determine which three images are relevant vs. irrelevant,
      - for each anchor image of the requested type (relevant/irrelevant),
        rotate all trial fixations so that this anchor's wedge is at the
        canonical top position,
      - split each fixation's duration evenly across the anchors of that type
        within the trial to avoid triple-counting.

        Returns a DataFrame with columns at least:
            subject, game, trial_number, anchor_image, anchor_type,
            time_rel, x_rot, y_rot, duration

        If anchor_valence_filter is provided (e.g., ("positive",) or
        ("negative",)), anchors within a trial are further restricted to
        images whose valence is in that set before determining n_anchors.
    """

    if anchor_type not in ("relevant", "irrelevant"):
        raise ValueError("anchor_type must be 'relevant' or 'irrelevant'")

    fix_df = _load_choice_fixation_dataframe(subj, output_base_dir, buffer_size, roi_type, data_dir=data_dir)

    # Project root (for data/<subj>/...), inferred from task_path
    root_dir = os.path.dirname(task_path)
    if root_dir == "":
        root_dir = "."
    valence_map = _build_valence_map_for_choice(subj, root_dir)

    # Restrict to choice phase fixations
    choice_fix = fix_df[(fix_df["phase"] == "choice") & (fix_df["event"] == "choice")].copy()
    if choice_fix.empty:
        return pd.DataFrame()

    # Onset relative to the start of the choice recording for that trial
    choice_fix["fix_onset_relative"] = choice_fix["fix_start"] - choice_fix["eyetracker_onset"]

    # Trial grouping keys
    for col in ("game", "trial_number"):
        if col in choice_fix.columns:
            choice_fix[col] = pd.to_numeric(choice_fix[col], errors="coerce")

    games, image_positions = _load_game_and_position_info(subj, task_path)

    # Screen geometry (same as recall wedge script)
    center_x = 3840 / 2.0
    center_y = 2160 / 2.0
    circle_radius = 380
    _positions, angles = generate_circle_positions(radius=circle_radius, n_positions=6)

    # We keep the full choice window; time_rel is simply ms since choice onset
    choice_fix["time_rel"] = choice_fix["fix_onset_relative"]

    all_rows: list[dict] = []

    grouped = choice_fix.groupby(["game", "trial_number"], sort=True)
    for (game, trial), grp in grouped:
        if pd.isna(game) or pd.isna(trial):
            continue
        game_idx = int(game) - 1
        if game_idx < 0 or game_idx >= len(games):
            continue

        images_in_game = games[game_idx]

        # Decision label (take/leave) for this trial, if available
        decision_label = None
        if "choice" in grp.columns:
            dec_vals = grp["choice"].dropna().unique()
            if len(dec_vals):
                try:
                    dec_int = int(dec_vals[0])
                except Exception:
                    dec_int = None
                if dec_int == 1:
                    decision_label = "take"
                elif dec_int == 2:
                    decision_label = "leave"

        # Determine option token for this trial
        option_token = None
        if "option" in grp.columns:
            opts = grp["option"].dropna().astype(str).unique()
            if len(opts):
                option_token = opts[0]

        if not option_token:
            # If we don't know which feature was relevant, skip this trial
            continue

        # Determine which images are relevant vs. irrelevant in this trial
        relevant_images: list[str] = []
        irrelevant_images: list[str] = []
        for img in images_in_game:
            if option_token in img.split("_") or option_token in img:
                relevant_images.append(img)
            else:
                irrelevant_images.append(img)

        if anchor_type == "relevant":
            anchors = relevant_images
        else:
            anchors = irrelevant_images

        # Optionally restrict anchors by valence (e.g., positive-only)
        if anchor_valence_filter is not None:
            anchors = [
                img
                for img in anchors
                if valence_map.get((int(game), img), "neutral") in anchor_valence_filter
            ]

        n_anchors = len(anchors)
        if n_anchors == 0:
            continue

        # Only consider fixations on item locations; exclude center and 'none'
        valid = (
            grp["roi_content"].notna()
            & (grp["roi_content"] != "none")
            & (grp["roi_content"] != "fixation")
        )
        trial_fix = grp[valid].copy()
        if trial_fix.empty:
            continue

        # Each fixation's contribution will be split evenly across anchors of this type
        duration_col = "fix_duration_bounded" if "fix_duration_bounded" in trial_fix.columns else "fix_duration_full"

        for anchor_img in anchors:
            if anchor_img not in image_positions:
                # If image is missing from the positions map, skip this anchor
                continue
            target_pos_index = image_positions[anchor_img]
            try:
                target_pos_index = int(target_pos_index)
            except Exception:
                continue
            if target_pos_index < 0 or target_pos_index >= len(angles):
                continue

            target_angle = angles[target_pos_index]
            canonical_angle = angles[0]  # top (90 degrees)
            rotation_deg = canonical_angle - target_angle
            rotation_rad = np.deg2rad(rotation_deg)

            anchor_valence = valence_map.get((int(game), anchor_img), "neutral")

            for _, fix in trial_fix.iterrows():
                time_rel = float(fix["time_rel"])

                # Convert fixation to math-style coordinates (x right, y up)
                fx = float(fix["fix_x"]) - center_x
                fy = center_y - float(fix["fix_y"])

                x_rot = fx * np.cos(rotation_rad) - fy * np.sin(rotation_rad)
                y_rot = fx * np.sin(rotation_rad) + fy * np.cos(rotation_rad)

                dur = float(fix[duration_col]) / n_anchors
                if dur <= 0:
                    continue

                all_rows.append(
                    {
                        "subject": subj,
                        "game": int(game),
                        "trial_number": int(trial),
                        "anchor_image": anchor_img,
                        "anchor_type": anchor_type,
                        "anchor_valence": anchor_valence,
                        "decision_label": decision_label,
                        "time_rel": time_rel,
                        "x_rot": float(x_rot),
                        "y_rot": float(y_rot),
                        "duration": dur,
                    }
                )

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


def collect_choice_rotated_fixations_valence(
    subj: str,
    output_base_dir: str,
    task_path: str,
    buffer_size: int,
    roi_type: str,
    valence_filter: tuple[str, ...],
    data_dir: str | None = None,
) -> pd.DataFrame:
    """Collect wedge-aligned choice fixations for one subject, for a
    specified set of item valences (e.g., all positive or all negative).

    Here we ignore relevance when choosing anchors: all images in the game
    whose valence is in ``valence_filter`` are treated as anchors. For each
    such anchor, we rotate the full trial so that this image is at the
    canonical top position and split each fixation's duration evenly across
    the anchors of that valence on that trial.

    Returned rows are labeled with the anchor's relevance (relevant vs
    irrelevant), valence, and the trial's decision_label so that later
    steps can subset by these factors.
    """

    fix_df = _load_choice_fixation_dataframe(subj, output_base_dir, buffer_size, roi_type, data_dir=data_dir)

    # Project root (for data/<subj>/...), inferred from task_path
    root_dir = os.path.dirname(task_path)
    if root_dir == "":
        root_dir = "."
    valence_map = _build_valence_map_for_choice(subj, root_dir)

    choice_fix = fix_df[(fix_df["phase"] == "choice") & (fix_df["event"] == "choice")].copy()
    if choice_fix.empty:
        return pd.DataFrame()

    choice_fix["fix_onset_relative"] = choice_fix["fix_start"] - choice_fix["eyetracker_onset"]

    for col in ("game", "trial_number"):
        if col in choice_fix.columns:
            choice_fix[col] = pd.to_numeric(choice_fix[col], errors="coerce")

    games, image_positions = _load_game_and_position_info(subj, task_path)

    center_x = 3840 / 2.0
    center_y = 2160 / 2.0
    circle_radius = 380
    _positions, angles = generate_circle_positions(radius=circle_radius, n_positions=6)

    choice_fix["time_rel"] = choice_fix["fix_onset_relative"]

    all_rows: list[dict] = []

    grouped = choice_fix.groupby(["game", "trial_number"], sort=True)
    for (game, trial), grp in grouped:
        if pd.isna(game) or pd.isna(trial):
            continue
        game_idx = int(game) - 1
        if game_idx < 0 or game_idx >= len(games):
            continue

        images_in_game = games[game_idx]

        # Decision label (take/leave) for this trial, if available
        decision_label = None
        if "choice" in grp.columns:
            dec_vals = grp["choice"].dropna().unique()
            if len(dec_vals):
                try:
                    dec_int = int(dec_vals[0])
                except Exception:
                    dec_int = None
                if dec_int == 1:
                    decision_label = "take"
                elif dec_int == 2:
                    decision_label = "leave"

        # Determine option token for this trial (for relevance labeling)
        option_token = None
        if "option" in grp.columns:
            opts = grp["option"].dropna().astype(str).unique()
            if len(opts):
                option_token = opts[0]

        if not option_token:
            continue

        relevant_images: list[str] = []
        irrelevant_images: list[str] = []
        for img in images_in_game:
            if option_token in img.split("_") or option_token in img:
                relevant_images.append(img)
            else:
                irrelevant_images.append(img)

        # Anchors are all images of the requested valence(s), regardless of relevance
        anchors: list[str] = []
        for img in images_in_game:
            if valence_map.get((int(game), img), "neutral") in valence_filter:
                anchors.append(img)

        n_anchors = len(anchors)
        if n_anchors == 0:
            continue

        valid = (
            grp["roi_content"].notna()
            & (grp["roi_content"] != "none")
            & (grp["roi_content"] != "fixation")
        )
        trial_fix = grp[valid].copy()
        if trial_fix.empty:
            continue

        duration_col = "fix_duration_bounded" if "fix_duration_bounded" in trial_fix.columns else "fix_duration_full"

        for anchor_img in anchors:
            if anchor_img not in image_positions:
                continue
            target_pos_index = image_positions[anchor_img]
            try:
                target_pos_index = int(target_pos_index)
            except Exception:
                continue
            if target_pos_index < 0 or target_pos_index >= len(angles):
                continue

            target_angle = angles[target_pos_index]
            canonical_angle = angles[0]
            rotation_deg = canonical_angle - target_angle
            rotation_rad = np.deg2rad(rotation_deg)

            anchor_valence = valence_map.get((int(game), anchor_img), "neutral")
            anchor_type = "relevant" if anchor_img in relevant_images else "irrelevant"

            for _, fix in trial_fix.iterrows():
                time_rel = float(fix["time_rel"])

                fx = float(fix["fix_x"]) - center_x
                fy = center_y - float(fix["fix_y"])

                x_rot = fx * np.cos(rotation_rad) - fy * np.sin(rotation_rad)
                y_rot = fx * np.sin(rotation_rad) + fy * np.cos(rotation_rad)

                dur = float(fix[duration_col]) / n_anchors
                if dur <= 0:
                    continue

                all_rows.append(
                    {
                        "subject": subj,
                        "game": int(game),
                        "trial_number": int(trial),
                        "anchor_image": anchor_img,
                        "anchor_type": anchor_type,
                        "anchor_valence": anchor_valence,
                        "decision_label": decision_label,
                        "time_rel": time_rel,
                        "x_rot": float(x_rot),
                        "y_rot": float(y_rot),
                        "duration": dur,
                    }
                )

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


def get_choice_valence_contrast_heatmap_data(
    output_base_dir: str,
    task_path: str,
    roi_type: str,
    buffer_size: int,
    anchor_type: str,
    decision_label: str,
    outer_radius: int = 530,
    central_radius: int = 20,
    circle_radius: int = 380,
):
    """Compute group-level positive-minus-negative contrast for a given
    anchor type (relevant/irrelevant) and decision (take/leave).

    For each subject we build separate normalized maps for *positive* and
    *negative* anchors under the requested condition, then form a
    subject-level difference map:

        D_s = H_pos_s - H_neg_s

    The group map is the mean of these differences across subjects, with a
    final Gaussian smoothing.
    """

    if anchor_type not in ("relevant", "irrelevant"):
        raise ValueError("anchor_type must be 'relevant' or 'irrelevant'")
    if decision_label not in ("take", "leave"):
        raise ValueError("decision_label must be 'take' or 'leave'")

    subjects = _find_subjects_with_fixations(output_base_dir)
    if not subjects:
        print("No subjects with fixation data found; cannot build valence contrast heatmap.")
        return None

    x_range = (-outer_radius - 5, outer_radius + 5)
    y_range = (-outer_radius - 5, outer_radius + 5)
    n_bins_x = 50
    n_bins_y = 50

    subject_diffs = []
    contributing_subjects: list[str] = []

    for subj in subjects:
        total_fix_time = _get_total_item_fix_time(subj, output_base_dir, buffer_size, roi_type)
        if total_fix_time <= 0:
            continue
        try:
            rotated_pos = collect_choice_rotated_fixations_valence(
                subj=subj,
                output_base_dir=output_base_dir,
                task_path=task_path,
                buffer_size=buffer_size,
                roi_type=roi_type,
                valence_filter=("positive",),
            )
            rotated_neg = collect_choice_rotated_fixations_valence(
                subj=subj,
                output_base_dir=output_base_dir,
                task_path=task_path,
                buffer_size=buffer_size,
                roi_type=roi_type,
                valence_filter=("negative",),
            )
        except FileNotFoundError as e:
            print(f"[WARN] Skipping subject {subj} for valence contrast map: {e}")
            continue
        except Exception as e:
            print(f"[WARN] Failed to collect rotated fixations for {subj} (valence contrast): {e}")
            continue

        if rotated_pos.empty or rotated_neg.empty:
            continue

        if "decision_label" not in rotated_pos.columns or "decision_label" not in rotated_neg.columns:
            continue

        # Condition on decision and requested relevance (anchor_type)
        df_pos = rotated_pos[
            (rotated_pos["decision_label"] == decision_label)
            & (rotated_pos["anchor_type"] == anchor_type)
        ].copy()
        df_neg = rotated_neg[
            (rotated_neg["decision_label"] == decision_label)
            & (rotated_neg["anchor_type"] == anchor_type)
        ].copy()
        if df_pos.empty or df_neg.empty:
            continue

        def _subject_map(df: pd.DataFrame) -> np.ndarray | None:
            durations = df["duration"].to_numpy()
            if durations.sum() <= 0:
                return None
            H, _xe, _ye = np.histogram2d(
                df["x_rot"].to_numpy(),
                df["y_rot"].to_numpy(),
                bins=[n_bins_x, n_bins_y],
                range=[x_range, y_range],
                weights=durations,
            )
            if H.sum() <= 0:
                return None
            # Normalize by total item fixation time so this map encodes
            # proportion of *total* fixation time in each bin.
            return H / total_fix_time

        H_pos = _subject_map(df_pos)
        H_neg = _subject_map(df_neg)
        if H_pos is None or H_neg is None:
            continue

        D = H_pos - H_neg
        subject_diffs.append(D)
        contributing_subjects.append(subj)

    if not subject_diffs:
        print(
            f"No subject contributed data to the valence contrast heatmap "
            f"for anchor_type={anchor_type}, decision_label={decision_label}."
        )
        return None

    group_D = np.mean(np.stack(subject_diffs, axis=0), axis=0)
    if np.any(group_D):
        group_D_smooth = gaussian_filter(group_D, sigma=2.0)
    else:
        group_D_smooth = group_D

    return {
        "group_D_smooth": group_D_smooth,
        "x_range": x_range,
        "y_range": y_range,
        "outer_radius": outer_radius,
        "central_radius": central_radius,
        "circle_radius": circle_radius,
        "n_subjects": len(contributing_subjects),
        "subjects": contributing_subjects,
        "anchor_type": anchor_type,
        "decision_label": decision_label,
    }


def get_choice_decision_contrast_heatmap_data(
    output_base_dir: str,
    task_path: str,
    roi_type: str,
    buffer_size: int,
    anchor_type: str,
    anchor_valence: str,
    outer_radius: int = 530,
    central_radius: int = 20,
    circle_radius: int = 380,
):
    """Compute group-level take-minus-leave contrast for a given
    anchor_type (relevant/irrelevant) and anchor_valence (positive/negative).

    For each subject we build separate normalized maps for *take* and
    *leave* decisions under the requested (relevance, valence) condition,
    then form a subject-level difference map:

        D_s = H_take_s - H_leave_s

    The group map is the mean of these differences across subjects, with a
    final Gaussian smoothing.
    """

    if anchor_type not in ("relevant", "irrelevant"):
        raise ValueError("anchor_type must be 'relevant' or 'irrelevant'")
    if anchor_valence not in ("positive", "negative"):
        raise ValueError("anchor_valence must be 'positive' or 'negative'")

    subjects = _find_subjects_with_fixations(output_base_dir)
    if not subjects:
        print("No subjects with fixation data found; cannot build decision contrast heatmap.")
        return None

    x_range = (-outer_radius - 5, outer_radius + 5)
    y_range = (-outer_radius - 5, outer_radius + 5)
    n_bins_x = 50
    n_bins_y = 50

    subject_diffs: list[np.ndarray] = []
    contributing_subjects: list[str] = []

    for subj in subjects:
        total_fix_time = _get_total_item_fix_time(subj, output_base_dir, buffer_size, roi_type)
        if total_fix_time <= 0:
            continue
        try:
            rotated = collect_choice_rotated_fixations_valence(
                subj=subj,
                output_base_dir=output_base_dir,
                task_path=task_path,
                buffer_size=buffer_size,
                roi_type=roi_type,
                valence_filter=(anchor_valence,),
            )
        except FileNotFoundError as e:
            print(f"[WARN] Skipping subject {subj} for decision contrast map: {e}")
            continue
        except Exception as e:
            print(f"[WARN] Failed to collect rotated fixations for {subj} (decision contrast): {e}")
            continue

        if rotated.empty:
            continue

        if "decision_label" not in rotated.columns or "anchor_type" not in rotated.columns:
            continue

        df_cond = rotated[rotated["anchor_type"] == anchor_type].copy()
        if df_cond.empty:
            continue

        df_take = df_cond[df_cond["decision_label"] == "take"].copy()
        df_leave = df_cond[df_cond["decision_label"] == "leave"].copy()
        if df_take.empty or df_leave.empty:
            continue

        def _subject_map(df: pd.DataFrame) -> np.ndarray | None:
            durations = df["duration"].to_numpy()
            if durations.sum() <= 0:
                return None
            H, _xe, _ye = np.histogram2d(
                df["x_rot"].to_numpy(),
                df["y_rot"].to_numpy(),
                bins=[n_bins_x, n_bins_y],
                range=[x_range, y_range],
                weights=durations,
            )
            if H.sum() <= 0:
                return None
            return H / total_fix_time

        H_take = _subject_map(df_take)
        H_leave = _subject_map(df_leave)
        if H_take is None or H_leave is None:
            continue

        D = H_take - H_leave
        subject_diffs.append(D)
        contributing_subjects.append(subj)

    if not subject_diffs:
        print(
            f"No subject contributed data to the decision contrast heatmap "
            f"for anchor_type={anchor_type}, anchor_valence={anchor_valence}."
        )
        return None

    group_D = np.mean(np.stack(subject_diffs, axis=0), axis=0)
    if np.any(group_D):
        group_D_smooth = gaussian_filter(group_D, sigma=2.0)
    else:
        group_D_smooth = group_D

    return {
        "group_D_smooth": group_D_smooth,
        "x_range": x_range,
        "y_range": y_range,
        "outer_radius": outer_radius,
        "central_radius": central_radius,
        "circle_radius": circle_radius,
        "n_subjects": len(contributing_subjects),
        "subjects": contributing_subjects,
        "anchor_type": anchor_type,
        "anchor_valence": anchor_valence,
    }


def get_choice_interaction_heatmap_data(
    output_base_dir: str,
    task_path: str,
    roi_type: str,
    buffer_size: int,
    anchor_type: str,
    outer_radius: int = 530,
    central_radius: int = 20,
    circle_radius: int = 380,
    data_dir: str | None = None,
):
    """Compute group-level interaction contrast for a given anchor_type.

    For each subject, we use the relevant/irrelevant anchors to build four
    normalized maps:

        H_take_pos   : decision = take,   valence = positive
        H_take_neg   : decision = take,   valence = negative
        H_leave_pos  : decision = leave,  valence = positive
        H_leave_neg  : decision = leave,  valence = negative

    all restricted to the requested anchor_type (relevant/irrelevant).

    The subject-level interaction map is then the difference-of-differences:

        I_s = (H_take_pos - H_take_neg) - (H_leave_pos - H_leave_neg)

    The group map is the mean of these interaction maps across subjects,
    followed by Gaussian smoothing.
    """

    if anchor_type not in ("relevant", "irrelevant"):
        raise ValueError("anchor_type must be 'relevant' or 'irrelevant'")

    subjects = _find_subjects_with_fixations(output_base_dir, data_dir=data_dir)
    if not subjects:
        print("No subjects with fixation data found; cannot build interaction heatmap.")
        return None

    x_range = (-outer_radius - 5, outer_radius + 5)
    y_range = (-outer_radius - 5, outer_radius + 5)
    n_bins_x = 50
    n_bins_y = 50

    subject_interactions: list[np.ndarray] = []
    contributing_subjects: list[str] = []

    for subj in subjects:
        total_fix_time = _get_total_item_fix_time(subj, output_base_dir, buffer_size, roi_type, data_dir=data_dir)
        if total_fix_time <= 0:
            continue
        try:
            rotated_pos = collect_choice_rotated_fixations_valence(
                subj=subj,
                output_base_dir=output_base_dir,
                task_path=task_path,
                buffer_size=buffer_size,
                roi_type=roi_type,
                valence_filter=("positive",),
                data_dir=data_dir,
            )
            rotated_neg = collect_choice_rotated_fixations_valence(
                subj=subj,
                output_base_dir=output_base_dir,
                task_path=task_path,
                buffer_size=buffer_size,
                roi_type=roi_type,
                valence_filter=("negative",),
                data_dir=data_dir,
            )
        except FileNotFoundError as e:
            print(f"[WARN] Skipping subject {subj} for interaction map: {e}")
            continue
        except Exception as e:
            print(f"[WARN] Failed to collect rotated fixations for {subj} (interaction): {e}")
            continue

        if rotated_pos.empty or rotated_neg.empty:
            continue

        if "decision_label" not in rotated_pos.columns or "anchor_type" not in rotated_pos.columns:
            continue
        if "decision_label" not in rotated_neg.columns or "anchor_type" not in rotated_neg.columns:
            continue

        # Restrict to requested relevance
        pos_rel = rotated_pos[rotated_pos["anchor_type"] == anchor_type].copy()
        neg_rel = rotated_neg[rotated_neg["anchor_type"] == anchor_type].copy()
        if pos_rel.empty or neg_rel.empty:
            continue

        # Split by decision and valence
        take_pos = pos_rel[pos_rel["decision_label"] == "take"].copy()
        leave_pos = pos_rel[pos_rel["decision_label"] == "leave"].copy()
        take_neg = neg_rel[neg_rel["decision_label"] == "take"].copy()
        leave_neg = neg_rel[neg_rel["decision_label"] == "leave"].copy()
        if take_pos.empty or leave_pos.empty or take_neg.empty or leave_neg.empty:
            continue

        def _subject_map(df: pd.DataFrame) -> np.ndarray | None:
            durations = df["duration"].to_numpy()
            if durations.sum() <= 0:
                return None
            H, _xe, _ye = np.histogram2d(
                df["x_rot"].to_numpy(),
                df["y_rot"].to_numpy(),
                bins=[n_bins_x, n_bins_y],
                range=[x_range, y_range],
                weights=durations,
            )
            if H.sum() <= 0:
                return None
            return H / total_fix_time

        H_take_pos = _subject_map(take_pos)
        H_take_neg = _subject_map(take_neg)
        H_leave_pos = _subject_map(leave_pos)
        H_leave_neg = _subject_map(leave_neg)
        if any(h is None for h in (H_take_pos, H_take_neg, H_leave_pos, H_leave_neg)):
            continue

        I = (H_take_pos - H_take_neg) - (H_leave_pos - H_leave_neg)
        subject_interactions.append(I)
        contributing_subjects.append(subj)

    if not subject_interactions:
        print(f"No subject contributed data to the interaction heatmap for anchor_type={anchor_type}.")
        return None

    group_I = np.mean(np.stack(subject_interactions, axis=0), axis=0)
    if np.any(group_I):
        group_I_smooth = gaussian_filter(group_I, sigma=2.0)
    else:
        group_I_smooth = group_I

    return {
        "group_D_smooth": group_I_smooth,
        "x_range": x_range,
        "y_range": y_range,
        "outer_radius": outer_radius,
        "central_radius": central_radius,
        "circle_radius": circle_radius,
        "n_subjects": len(contributing_subjects),
        "subjects": contributing_subjects,
        "anchor_type": anchor_type,
    }


def get_choice_heatmap_data(
    output_base_dir: str,
    task_path: str,
    roi_type: str,
    buffer_size: int,
    anchor_type: str,
    outer_radius: int = 530,
    central_radius: int = 20,
    circle_radius: int = 380,
):
    """Compute group-level wedge heatmap for the choice task.

    This mirrors get_cluster_heatmap_data for recall, but collapses across
    the full choice period and conditions on anchor_type (relevant vs. irrelevant).

    Returns a dict with keys:
      group_H_smooth, x_range, y_range, outer_radius, central_radius, circle_radius.
    Returns None if no valid data.
    """

    subjects = _find_subjects_with_fixations(output_base_dir)
    if not subjects:
        print("No subjects with fixation data found; cannot build choice heatmap.")
        return None

    # Build or load per-subject rotated fixations and aggregate into subject maps
    x_range = (-outer_radius - 5, outer_radius + 5)
    y_range = (-outer_radius - 5, outer_radius + 5)
    n_bins_x = 50
    n_bins_y = 50

    subject_maps = []
    contributing_subjects: list[str] = []

    for subj in subjects:
        try:
            rotated_df = collect_choice_rotated_fixations(
                subj=subj,
                output_base_dir=output_base_dir,
                task_path=task_path,
                buffer_size=buffer_size,
                roi_type=roi_type,
                anchor_type=anchor_type,
            )
        except FileNotFoundError as e:
            print(f"[WARN] Skipping subject {subj}: {e}")
            continue
        except Exception as e:
            print(f"[WARN] Failed to collect rotated fixations for {subj}: {e}")
            continue

        if rotated_df.empty:
            continue

        durations = rotated_df["duration"].to_numpy()
        if durations.sum() <= 0:
            continue

        H, xedges, yedges = np.histogram2d(
            rotated_df["x_rot"].to_numpy(),
            rotated_df["y_rot"].to_numpy(),
            bins=[n_bins_x, n_bins_y],
            range=[x_range, y_range],
            weights=durations,
        )

        total = H.sum()
        if total <= 0:
            continue

        H_norm = H / total
        subject_maps.append(H_norm)
        contributing_subjects.append(subj)

    if not subject_maps:
        print("No subject contributed data to the choice heatmap.")
        return None

    group_H = np.mean(np.stack(subject_maps, axis=0), axis=0)
    if np.any(group_H):
        group_H_smooth = gaussian_filter(group_H, sigma=2.0)
    else:
        group_H_smooth = group_H

    return {
        "group_H_smooth": group_H_smooth,
        "x_range": x_range,
        "y_range": y_range,
        "outer_radius": outer_radius,
        "central_radius": central_radius,
        "circle_radius": circle_radius,
        "n_subjects": len(contributing_subjects),
        "subjects": contributing_subjects,
    }


def get_choice_contrast_heatmap_data(
    output_base_dir: str,
    task_path: str,
    roi_type: str,
    buffer_size: int,
    outer_radius: int = 530,
    central_radius: int = 20,
    circle_radius: int = 380,
    data_dir: str | None = None,
):
    """Compute group-level relevant-minus-irrelevant wedge contrast for the choice task.

    For each subject we build separate normalized maps for relevant and
    irrelevant anchors, then form a subject-level difference map:

        D_s = H_rel_s - H_irr_s

    The group map is the mean of these differences across subjects, with a
    final Gaussian smoothing. This keeps the contrast at the subject level
    rather than differencing already-averaged maps.
    """

    subjects = _find_subjects_with_fixations(output_base_dir, data_dir=data_dir)
    if not subjects:
        print("No subjects with fixation data found; cannot build choice contrast heatmap.")
        return None

    x_range = (-outer_radius - 5, outer_radius + 5)
    y_range = (-outer_radius - 5, outer_radius + 5)
    n_bins_x = 50
    n_bins_y = 50

    subject_diffs = []
    contributing_subjects: list[str] = []

    for subj in subjects:
        try:
            rel_df = collect_choice_rotated_fixations(
                subj=subj,
                output_base_dir=output_base_dir,
                task_path=task_path,
                buffer_size=buffer_size,
                roi_type=roi_type,
                anchor_type="relevant",
                data_dir=data_dir,
            )
            irr_df = collect_choice_rotated_fixations(
                subj=subj,
                output_base_dir=output_base_dir,
                task_path=task_path,
                buffer_size=buffer_size,
                roi_type=roi_type,
                anchor_type="irrelevant",
                data_dir=data_dir,
            )
        except FileNotFoundError as e:
            print(f"[WARN] Skipping subject {subj} for contrast map: {e}")
            continue
        except Exception as e:
            print(f"[WARN] Failed to collect rotated fixations for {subj} (contrast): {e}")
            continue

        if rel_df.empty or irr_df.empty:
            continue

        def _subject_map(df: pd.DataFrame) -> np.ndarray | None:
            durations = df["duration"].to_numpy()
            if durations.sum() <= 0:
                return None
            H, _xe, _ye = np.histogram2d(
                df["x_rot"].to_numpy(),
                df["y_rot"].to_numpy(),
                bins=[n_bins_x, n_bins_y],
                range=[x_range, y_range],
                weights=durations,
            )
            total = H.sum()
            if total <= 0:
                return None
            return H / total

        H_rel = _subject_map(rel_df)
        H_irr = _subject_map(irr_df)
        if H_rel is None or H_irr is None:
            continue

        D = H_rel - H_irr
        subject_diffs.append(D)
        contributing_subjects.append(subj)

    if not subject_diffs:
        print("No subject contributed data to the choice contrast heatmap.")
        return None

    group_D = np.mean(np.stack(subject_diffs, axis=0), axis=0)
    if np.any(group_D):
        group_D_smooth = gaussian_filter(group_D, sigma=2.0)
    else:
        group_D_smooth = group_D

    return {
        "group_D_smooth": group_D_smooth,
        "x_range": x_range,
        "y_range": y_range,
        "outer_radius": outer_radius,
        "central_radius": central_radius,
        "circle_radius": circle_radius,
        "n_subjects": len(contributing_subjects),
        "subjects": contributing_subjects,
    }


def plot_choice_heatmap(
    data: dict,
    anchor_type: str,
    out_png: str,
    out_pdf: str | None = None,
) -> None:
    """Render a single choice-task wedge heatmap using the recall-style styling."""

    if data is None:
        return

    group_H_smooth = data["group_H_smooth"]
    x_range = data["x_range"]
    y_range = data["y_range"]
    outer_radius = data["outer_radius"]
    central_radius = data["central_radius"]
    circle_radius = data["circle_radius"]

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    gray_hot = sns.color_palette("BuPu", as_cmap=True)

    extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
    im = ax.imshow(
        group_H_smooth.T,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=gray_hot,
        vmin=0,
    )

    clip_circle = Circle((0, 0), outer_radius, transform=ax.transData)
    im.set_clip_path(clip_circle)

    cax = inset_axes(
        ax,
        width="3%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    cbar = plt.colorbar(im, cax=cax)
    data_max = float(np.nanmax(group_H_smooth)) if np.any(group_H_smooth) else 0.0
    im.set_clim(0.0, data_max)
    if data_max > 0:
        ticks = np.linspace(0.0, 0.0025, 6)
    else:
        ticks = np.array([0.0])
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t * 1e3:g}" for t in ticks])
    cbar.set_label("Mean proportion fixation time (×10⁻³)")

    draw_full_roi(ax, outer_radius=outer_radius, central_radius=central_radius)

    rect_size = 160
    half_size = rect_size / 2.0
    rect_center_x = 0.0
    rect_center_y = circle_radius
    rect = Rectangle(
        (rect_center_x - half_size, rect_center_y - half_size),
        rect_size,
        rect_size,
        fill=False,
        edgecolor="black",
        linestyle="--",
        linewidth=2,
    )
    ax.add_patch(rect)

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    title_anchor = "Relevant" if anchor_type == "relevant" else "Irrelevant"
    ax.set_title(f"{title_anchor} items at top (choice)", fontsize=18, pad=10)

    fig.tight_layout(rect=(0, 0, 0.85, 1))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300)
    if out_pdf is not None:
        fig.savefig(out_pdf)
    plt.close(fig)


def plot_choice_contrast_heatmap(
    data: dict,
    out_png: str,
    out_pdf: str | None = None,
    title: str | None = None,
    cbar_label: str | None = None,
    vmin_override: float | None = None,
    vmax_override: float | None = None,
) -> None:
    """Render a generic contrast heatmap (zero-centered diverging colormap).

    For the relevance contrast, positive values indicate greater fixation
    proportion for relevant than irrelevant items. For the valence
    contrasts, positive values indicate greater fixation proportion for
    positive than negative items.
    """

    if data is None:
        return

    group_D_smooth = data["group_D_smooth"]
    x_range = data["x_range"]
    y_range = data["y_range"]
    outer_radius = data["outer_radius"]
    central_radius = data["central_radius"]
    circle_radius = data["circle_radius"]

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    diverge = sns.diverging_palette(240, 10, as_cmap=True)

    extent = [x_range[0], x_range[1], y_range[0], y_range[1]]

    # Determine color limits: either based on data or an explicit override
    max_abs = float(np.nanmax(np.abs(group_D_smooth))) if np.any(group_D_smooth) else 0.0
    if vmin_override is not None and vmax_override is not None:
        vmin, vmax = vmin_override, vmax_override
    else:
        if max_abs == 0.0:
            vmin, vmax = -0.001, 0.001
        else:
            vmin, vmax = -max_abs, max_abs

    im = ax.imshow(
        group_D_smooth.T,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=diverge,
        vmin=vmin,
        vmax=vmax,
    )

    clip_circle = Circle((0, 0), outer_radius, transform=ax.transData)
    im.set_clip_path(clip_circle)

    cax = inset_axes(
        ax,
        width="3%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    cbar = plt.colorbar(im, cax=cax)
    # Use symmetric ticks around zero based on actual vmin/vmax
    if vmax_override is not None and vmin_override is not None:
        ticks = np.linspace(vmin, vmax, 7)
    elif max_abs > 0:
        ticks = np.linspace(-max_abs, max_abs, 7)
    else:
        ticks = np.array([0.0])
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t * 1e3:g}" for t in ticks])
    if cbar_label is None:
        cbar_label = "Relevant \u2022 Irrelevant fixation time (×10⁻³)"
    cbar.set_label(cbar_label)

    draw_full_roi(ax, outer_radius=outer_radius, central_radius=central_radius)

    rect_size = 160
    half_size = rect_size / 2.0
    rect_center_x = 0.0
    rect_center_y = circle_radius
    rect = Rectangle(
        (rect_center_x - half_size, rect_center_y - half_size),
        rect_size,
        rect_size,
        fill=False,
        edgecolor="black",
        linestyle="--",
        linewidth=2,
    )
    ax.add_patch(rect)

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    if title is None:
        title = "Relevant minus Irrelevant (choice)"
    ax.set_title(title, fontsize=18, pad=10)

    fig.tight_layout(rect=(0, 0, 0.85, 1))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300)
    if out_pdf is not None:
        fig.savefig(out_pdf)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create group-level wedge-aligned fixation heatmaps for the choice task, "
            "separately for relevant and irrelevant items, using styling matched to "
            "the recall cluster heatmap in analyze_eyetracking."
        )
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="data",
        help="Base data directory containing per-subject fixation CSVs (default: data)",
    )
    parser.add_argument(
        "--task_path",
        type=str,
        default="task",
        help="Base path to task files (must contain emdm-eyetracking/game_info)",
    )
    parser.add_argument(
        "--roi_type",
        type=str,
        default="original",
        choices=["original", "equal_area"],
        help="ROI type for fixation CSVs",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=50,
        help="Buffer size used in ROI processing (must match fixation CSVs)",
    )
    parser.add_argument(
        "--anchor_types",
        type=str,
        nargs="+",
        default=["relevant", "irrelevant"],
        choices=["relevant", "irrelevant"],
        help="Which anchor types to plot (default: both relevant and irrelevant)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="output/choice_wedge_heatmaps",
        help="Directory to save the choice wedge heatmap figures",
    )
    parser.add_argument(
        "--contrast",
        action="store_true",
        help="If set, also compute and plot the relevant-minus-irrelevant contrast heatmap.",
    )
    parser.add_argument(
        "--valence_contrast",
        action="store_true",
        help=(
            "If set, also compute and plot positive-minus-negative contrast heatmaps "
            "for each combination of decision (take/leave) and relevance (relevant/irrelevant)."
        ),
    )
    parser.add_argument(
        "--decision_contrast",
        action="store_true",
        help=(
            "If set, also compute and plot take-minus-leave contrast heatmaps "
            "for each combination of valence (positive/negative) and relevance (relevant/irrelevant)."
        ),
    )
    parser.add_argument(
        "--interaction_contrast",
        action="store_true",
        help=(
            "If set, also compute and plot interaction heatmaps capturing the "
            "valence × decision effect separately for relevant and irrelevant items."
        ),
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for anchor_type in args.anchor_types:
        data = get_choice_heatmap_data(
            output_base_dir=args.output_base_dir,
            task_path=args.task_path,
            roi_type=args.roi_type,
            buffer_size=args.buffer_size,
            anchor_type=anchor_type,
        )
        if data is None:
            continue

        base_name = f"group_choice_wedge_heatmap_{anchor_type}_{args.roi_type}_buffer_{args.buffer_size}"
        out_png = os.path.join(args.out_dir, base_name + ".png")
        out_pdf = os.path.join(args.out_dir, base_name + ".pdf")
        plot_choice_heatmap(data, anchor_type=anchor_type, out_png=out_png, out_pdf=out_pdf)

    if args.contrast:
        contrast_data = get_choice_contrast_heatmap_data(
            output_base_dir=args.output_base_dir,
            task_path=args.task_path,
            roi_type=args.roi_type,
            buffer_size=args.buffer_size,
        )
        if contrast_data is not None:
            base_name = f"group_choice_wedge_heatmap_contrast_{args.roi_type}_buffer_{args.buffer_size}"
            out_png = os.path.join(args.out_dir, base_name + ".png")
            out_pdf = os.path.join(args.out_dir, base_name + ".pdf")
            # Fix color scale so it matches interaction maps: ±0.0003
            plot_choice_contrast_heatmap(
                contrast_data,
                out_png=out_png,
                out_pdf=out_pdf,
                vmin_override=-0.0003,
                vmax_override=0.0003,
            )
    if args.valence_contrast:
        for anchor_type in ("relevant", "irrelevant"):
            for decision_label in ("take", "leave"):
                val_data = get_choice_valence_contrast_heatmap_data(
                    output_base_dir=args.output_base_dir,
                    task_path=args.task_path,
                    roi_type=args.roi_type,
                    buffer_size=args.buffer_size,
                    anchor_type=anchor_type,
                    decision_label=decision_label,
                )
                if val_data is None:
                    continue
                base_name = (
                    f"group_choice_wedge_heatmap_valence_{anchor_type}_"
                    f"{decision_label}_{args.roi_type}_buffer_{args.buffer_size}"
                )
                out_png = os.path.join(args.out_dir, base_name + ".png")
                out_pdf = os.path.join(args.out_dir, base_name + ".pdf")
                title = (
                    f"{anchor_type.capitalize()} items, {decision_label}: "
                    f"Positive minus Negative (choice)"
                )
                cbar_label = "Positive \u2022 Negative fixation time (×10⁻³)"
                plot_choice_contrast_heatmap(
                    val_data,
                    out_png=out_png,
                    out_pdf=out_pdf,
                    title=title,
                    cbar_label=cbar_label,
                )
    if args.decision_contrast:
        for anchor_type in ("relevant", "irrelevant"):
            for anchor_valence in ("positive", "negative"):
                dec_data = get_choice_decision_contrast_heatmap_data(
                    output_base_dir=args.output_base_dir,
                    task_path=args.task_path,
                    roi_type=args.roi_type,
                    buffer_size=args.buffer_size,
                    anchor_type=anchor_type,
                    anchor_valence=anchor_valence,
                )
                if dec_data is None:
                    continue
                base_name = (
                    f"group_choice_wedge_heatmap_decision_{anchor_type}_"
                    f"{anchor_valence}_{args.roi_type}_buffer_{args.buffer_size}"
                )
                out_png = os.path.join(args.out_dir, base_name + ".png")
                out_pdf = os.path.join(args.out_dir, base_name + ".pdf")
                title = (
                    f"{anchor_type.capitalize()} {anchor_valence} items: "
                    f"Take minus Leave (choice)"
                )
                cbar_label = "Take \u2022 Leave fixation time (×10⁻³)"
                plot_choice_contrast_heatmap(
                    dec_data,
                    out_png=out_png,
                    out_pdf=out_pdf,
                    title=title,
                    cbar_label=cbar_label,
                )

    if args.interaction_contrast:
        for anchor_type in ("relevant", "irrelevant"):
            int_data = get_choice_interaction_heatmap_data(
                output_base_dir=args.output_base_dir,
                task_path=args.task_path,
                roi_type=args.roi_type,
                buffer_size=args.buffer_size,
                anchor_type=anchor_type,
            )
            if int_data is None:
                continue
            base_name = (
                f"group_choice_wedge_heatmap_interaction_{anchor_type}_"
                f"{args.roi_type}_buffer_{args.buffer_size}"
            )
            out_png = os.path.join(args.out_dir, base_name + ".png")
            out_pdf = os.path.join(args.out_dir, base_name + ".pdf")
            title = (
                f"{anchor_type.capitalize()} items: "
                f"(Take pos−neg) minus (Leave pos−neg)"
            )
            cbar_label = "Interaction (valence × decision) fixation time (×10⁻³)"
            plot_choice_contrast_heatmap(
                int_data,
                out_png=out_png,
                out_pdf=out_pdf,
                title=title,
                cbar_label=cbar_label,
                vmin_override=-0.0003,
                vmax_override=0.0003,
            )


if __name__ == "__main__":
    main()
