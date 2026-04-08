#!/usr/bin/env python3
"""Compile NN simulation JSON into human-like fixation and behavioral CSVs.

Converts NN JSON files into per-subject fixation CSVs and behavioral logfiles
matching the human data layout (data/<SUBJ>/). Consecutive identical fixation
actions are collapsed into single rows.

Example:
  conda run -n analysis python metarnn/lib/compile_nn_to_human_fixations.py \\
    --json metarnn/simulations/data_0.json metarnn/simulations/data_1.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


FIXATION_COLUMNS: List[str] = [
    "phase",
    "event",
    "trial_number",
    "eyetracker_onset",
    "eyetracker_offset",
    "fix_start",
    "fix_end",
    "fix_duration_full",
    "fix_duration_bounded",
    "fix_x",
    "fix_y",
    "roi_x",
    "roi_y",
    "roi_content",
    "onset",
    "offset",
    "game",
    "image",
    "outcome",
    "choice",
    "correct",
    "rt",
    "letter",
    "match",
    "response",
    "option",
    "true_value",
    "true_position",
    "recalled_position",
    "Unnamed: 18",
]

# Write 'phase' without a UTF-8 BOM prefix so downstream scripts that
# expect a literal 'phase' column name work correctly.
LOGFILE_COLUMNS: List[str] = [
    "phase",
    "event",
    "onset",
    "offset",
    "trial_number",
    "game",
    "image",
    "outcome",
    "choice",
    "correct",
    "rt",
    "letter",
    "match",
    "response",
    "option",
    "true_value",
    "true_position",
    "recalled_position",
    "Unnamed: 18",
]


@dataclass(frozen=True)
class FeatureLabels:
    animacy_pos: str = "Animal"
    animacy_neg: str = "Object"
    env_pos: str = "Land"
    env_neg: str = "Sea"
    texture_pos: str = "Pattern"
    texture_neg: str = "Solid"
    size_pos: str = "Large"
    size_neg: str = "Small"

    def option_token(self, option_index: int) -> str:
        mapping = {
            0: self.animacy_pos,
            1: self.env_pos,
            2: self.texture_pos,
            3: self.size_pos,
        }
        if option_index not in mapping:
            raise ValueError(f"Unsupported options index {option_index}; expected 0-3.")
        return mapping[option_index]

    def item_name(self, bits: Sequence[int]) -> str:
        if len(bits) != 4:
            raise ValueError(f"Expected 4 feature bits, got {len(bits)}")
        a, e, t, s = (int(b) for b in bits)
        return "_".join(
            [
                self.animacy_pos if a == 1 else self.animacy_neg,
                self.env_pos if e == 1 else self.env_neg,
                self.texture_pos if t == 1 else self.texture_neg,
                self.size_pos if s == 1 else self.size_neg,
            ]
        )


def _roi_centers_px() -> Dict[int, Tuple[float, float]]:
    """Map fixation position index (0-5) to ROI center (x, y) in px."""

    return {
        0: (1920.0, 700.0),
        1: (2249.0, 891.0),
        2: (2249.0, 1269.0),
        3: (1920.0, 1460.0),
        4: (1591.0, 1269.0),
        5: (1591.0, 891.0),
    }


def load_nn_json(path: str) -> Dict[str, list]:
    with open(path, "r") as f:
        d = json.load(f)

    required = ["pairs", "features", "values", "options", "relevances", "offer_values", "actions"]
    for k in required:
        if k not in d:
            raise KeyError(f"Missing key {k} in {path}")

    n = len(d["actions"])
    for k in required:
        if len(d[k]) != n:
            raise ValueError(f"Length mismatch in {path}: {k} has len {len(d[k])} but actions has len {n}")

    return d


def compute_choice_and_rt(actions: Sequence[int], dt: float) -> Tuple[Optional[int], float]:
    """Return (choice_code, rt_seconds).

    choice_code matches the human coding used elsewhere in this repo:
      1 = take/accept
      2 = leave/reject

    If no terminal action (6/7) is found, returns (None, rt).
    """

    rt = float(len(actions)) * float(dt)

    choice_code = None
    # In the simulation spec, 6=accept, 7=reject.
    for a in reversed(actions):
        if a == 6:
            choice_code = 1
            break
        if a == 7:
            choice_code = 2
            break

    return choice_code, rt


def compute_correct(choice_code: Optional[int], offer_value: float) -> Optional[int]:
    """Correct if take positive offers and leave negative offers.

    Returns 1/0, or None if choice_code is missing or offer_value is not finite.
    """

    if choice_code not in (1, 2):
        return None
    try:
        v = float(offer_value)
    except Exception:
        return None
    if not np.isfinite(v) or v == 0:
        return None
    correct_choice = 1 if v > 0 else 2
    return 1 if choice_code == correct_choice else 0


def iter_fixation_segments(actions: Sequence[int]) -> Iterable[Tuple[int, int]]:
    """Yield (position_index, duration_timepoints) for collapsed fixation runs."""

    curr_pos: Optional[int] = None
    curr_dur = 0

    def flush():
        nonlocal curr_pos, curr_dur
        if curr_pos is not None and curr_dur > 0:
            yield (curr_pos, curr_dur)
        curr_pos, curr_dur = None, 0

    for a in actions:
        if a in (6, 7):
            # Decision made; stop.
            yield from flush()
            return
        if a < 0 or a > 5:
            # Unknown action; treat as break.
            yield from flush()
            continue
        if curr_pos is None:
            curr_pos = int(a)
            curr_dur = 1
        elif int(a) == curr_pos:
            curr_dur += 1
        else:
            yield from flush()
            curr_pos = int(a)
            curr_dur = 1

    yield from flush()


def build_logfile_rows(
    game_id: int,
    item_names: Sequence[str],
    item_values: Sequence[float],
    option_token: str,
    offer_value: float,
    choice_code: Optional[int],
    correct: Optional[int],
    rt: float,
) -> List[dict]:
    rows: List[dict] = []

    # Encoding image rows (needed by several analyses to reconstruct valence/outcome).
    for enc_trial, (nm, val) in enumerate(zip(item_names, item_values), start=1):
        rows.append(
            {
                "phase": "encoding",
                "event": "image",
                "onset": np.nan,
                "offset": np.nan,
                "trial_number": enc_trial,
                "game": game_id,
                "image": nm,
                "outcome": float(val),
                "choice": np.nan,
                "correct": np.nan,
                "rt": np.nan,
                "letter": np.nan,
                "match": np.nan,
                "response": np.nan,
                "option": np.nan,
                "true_value": np.nan,
                "true_position": np.nan,
                "recalled_position": np.nan,
                "Unnamed: 18": np.nan,
            }
        )

    # Choice row (one per simulated episode)
    rows.append(
        {
            "phase": "choice",
            "event": "choice",
            "onset": np.nan,
            "offset": np.nan,
            "trial_number": 1,
            "game": game_id,
            "image": np.nan,
            "outcome": np.nan,
            "choice": choice_code if choice_code is not None else np.nan,
            "correct": correct if correct is not None else np.nan,
            "rt": rt,
            "letter": np.nan,
            "match": np.nan,
            "response": np.nan,
            "option": option_token,
            "true_value": float(offer_value) if offer_value is not None else np.nan,
            "true_position": np.nan,
            "recalled_position": np.nan,
            "Unnamed: 18": np.nan,
        }
    )

    return rows


def build_fixation_rows_for_trial(
    game_id: int,
    item_by_pos: Dict[int, str],
    actions: Sequence[int],
    option_token: str,
    offer_value: float,
    choice_code: Optional[int],
    correct: Optional[int],
    rt: float,
    dt: float,
) -> List[dict]:
    roi_centers = _roi_centers_px()

    # Total trial duration in timepoints until decision (including the terminal decision action).
    total_timepoints = int(len(actions))

    # Collapsed fixation segments.
    rows: List[dict] = []
    t_ms = 0.0
    for pos, dur in iter_fixation_segments(actions):
        roi_x, roi_y = roi_centers.get(int(pos), (np.nan, np.nan))
        content = item_by_pos.get(int(pos), "none")
        fix_start = float(t_ms)
        fix_end = float(t_ms + dur)
        t_ms += dur

        rows.append(
            {
                "phase": "choice",
                "event": "choice",
                "trial_number": 1,
                "eyetracker_onset": 0,
                "eyetracker_offset": total_timepoints,
                "fix_start": fix_start,
                "fix_end": fix_end,
                "fix_duration_full": float(dur),
                "fix_duration_bounded": float(dur),
                "fix_x": float(roi_x),
                "fix_y": float(roi_y),
                "roi_x": float(roi_x),
                "roi_y": float(roi_y),
                "roi_content": content,
                "onset": 0.0,
                "offset": float(total_timepoints) * float(dt),
                "game": game_id,
                "image": np.nan,
                "outcome": np.nan,
                "choice": choice_code if choice_code is not None else np.nan,
                "correct": correct if correct is not None else np.nan,
                "rt": rt,
                "letter": np.nan,
                "match": np.nan,
                "response": np.nan,
                "option": option_token,
                "true_value": float(offer_value) if offer_value is not None else np.nan,
                "true_position": np.nan,
                "recalled_position": np.nan,
                "Unnamed: 18": np.nan,
            }
        )

    return rows


def compile_one_seed(
    json_path: str,
    subject_id: str,
    out_root: str,
    labels: FeatureLabels,
    dt: float,
    max_trials: Optional[int] = None,
) -> Tuple[str, str]:
    """Compile one JSON seed into a subject folder.

    Returns (fixations_csv_path, logfile_csv_path)
    """

    d = load_nn_json(json_path)
    n_trials = len(d["actions"])
    if max_trials is not None:
        n_trials = min(n_trials, int(max_trials))

    # Create target directories (both fixations and logfiles go under data/)
    data_dir = os.path.join(out_root, "data", subject_id)
    os.makedirs(data_dir, exist_ok=True)

    fixation_rows: List[dict] = []
    logfile_rows: List[dict] = []

    for i in range(n_trials):
        game_id = i + 1  # each episode gets its own "game" id for compatibility

        features = d["features"][i]
        values = d["values"][i]
        options = int(d["options"][i])
        offer_value = d["offer_values"][i]
        actions = d["actions"][i]

        # Build item names per position 0..5
        item_names: List[str] = [labels.item_name(bits) for bits in features]
        item_by_pos = {pos: nm for pos, nm in enumerate(item_names)}

        # Choice & RT
        choice_code, rt = compute_choice_and_rt(actions, dt=dt)
        correct = compute_correct(choice_code, offer_value)

        option_token = labels.option_token(options)

        logfile_rows.extend(
            build_logfile_rows(
                game_id=game_id,
                item_names=item_names,
                item_values=[float(v) for v in values],
                option_token=option_token,
                offer_value=float(offer_value),
                choice_code=choice_code,
                correct=correct,
                rt=rt,
            )
        )

        fixation_rows.extend(
            build_fixation_rows_for_trial(
                game_id=game_id,
                item_by_pos=item_by_pos,
                actions=actions,
                option_token=option_token,
                offer_value=float(offer_value),
                choice_code=choice_code,
                correct=correct,
                rt=rt,
                dt=dt,
            )
        )

    fix_df = pd.DataFrame(fixation_rows, columns=FIXATION_COLUMNS)
    log_df = pd.DataFrame(logfile_rows, columns=LOGFILE_COLUMNS)

    # Write outputs
    fix_path_buffer = os.path.join(data_dir, f"{subject_id}_fixations_df_original_buffer_50.csv")
    fix_path_nobuffername = os.path.join(data_dir, f"{subject_id}_fixations_df_original.csv")
    log_path = os.path.join(data_dir, f"{subject_id}_MAIN_logfile_7.csv")

    fix_df.to_csv(fix_path_buffer, index=False)
    fix_df.to_csv(fix_path_nobuffername, index=False)

    log_df.to_csv(log_path, index=False)

    return fix_path_buffer, log_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile NN simulation JSON files into human-like fixation/log CSVs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--json",
        nargs="+",
        required=True,
        help="One or more NN simulation JSON files (each treated as a separate seed/subject).",
    )
    parser.add_argument(
        "--out_root",
        default="metarnn/simulations/human_like",
        help="Output root directory to create data/ subfolder under.",
    )
    parser.add_argument(
        "--subject_start",
        type=int,
        default=901,
        help="First 3-digit synthetic subject id; increments by 1 per JSON.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.001,
        help="Seconds per action timepoint (purely a scaling convention for rt/onset/offset).",
    )
    parser.add_argument(
        "--max_trials",
        type=int,
        default=None,
        help="Optional cap on number of episodes/trials to compile from each JSON (debugging).",
    )

    args = parser.parse_args()

    labels = FeatureLabels()

    json_paths: List[str] = list(args.json)
    out_root: str = args.out_root
    os.makedirs(out_root, exist_ok=True)

    for j, jp in enumerate(json_paths):
        subject_id = f"{args.subject_start + j:03d}"
        fix_path, log_path = compile_one_seed(
            json_path=jp,
            subject_id=subject_id,
            out_root=out_root,
            labels=labels,
            dt=float(args.dt),
            max_trials=args.max_trials,
        )
        print(f"[OK] {jp} -> subject {subject_id}")
        print(f"     fixations: {fix_path}")
        print(f"     logfile:   {log_path}")


if __name__ == "__main__":
    main()
