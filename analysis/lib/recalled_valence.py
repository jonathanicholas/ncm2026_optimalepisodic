"""Shared helper to build valence maps from recalled (rather than true) values.

Used by choice_fixation_proportions.py and visualize_choice_fixation_wedges_group.py
to classify items as positive/negative based on what participants recalled,
falling back to the true outcome when recall is missing.
"""

import os

import numpy as np
import pandas as pd

POSITIVE_LABELS = {"PLUS", "+", "POSITIVE"}
NEGATIVE_LABELS = {"MINUS", "-", "NEGATIVE"}


def _parse_valuerecall(vr_df: pd.DataFrame, game: int) -> list:
    """Parse valuerecall rows for a single game into a list of recalled values.

    Returns a list of float or None (one per item in presentation order).
    None indicates the participant did not provide a valid recall.
    """
    rows = vr_df[vr_df["game"] == game].reset_index(drop=True)
    recalled_values = []
    i = 0
    while i < len(rows):
        label = str(rows.iloc[i]["item"]).upper().strip()
        if label == "NAN" or label == "" or label == "NONE":
            # No recall for this item
            recalled_values.append(None)
            i += 1
        elif label in POSITIVE_LABELS or label in NEGATIVE_LABELS:
            sign = 1 if label in POSITIVE_LABELS else -1
            if i + 1 < len(rows):
                try:
                    magnitude = int(float(rows.iloc[i + 1]["item"]))
                except (ValueError, TypeError):
                    magnitude = None
                if magnitude is not None:
                    recalled_values.append(sign * magnitude)
                else:
                    recalled_values.append(None)
                i += 2
            else:
                # Valence label without following magnitude
                recalled_values.append(None)
                i += 1
        else:
            # Unexpected label (shouldn't happen in well-formed data)
            recalled_values.append(None)
            i += 1
    return recalled_values


def build_recalled_valence_map(
    subj: str, data_dir: str
) -> dict[tuple, tuple[str, float]]:
    """Build {(game, image): (valence_str, value)} using recalled values.

    For each item, uses the participant's recalled value to determine valence.
    Falls back to the true outcome when recall is missing or unparseable.

    Parameters
    ----------
    subj : str
        Subject ID (e.g., '101').
    data_dir : str
        Path to directory containing subject folders (e.g., 'data/').
        Expects data_dir/SUBID/SUBID_MAIN_logfile_7.csv and
        data_dir/SUBID/SUBID_valuerecall.csv.

    Returns
    -------
    dict
        Mapping of (game, image) -> (valence_str, value) where valence_str
        is 'positive', 'negative', or 'neutral' and value is the numeric
        recalled value (or true value as fallback).
    """
    log_path = os.path.join(data_dir, subj, f"{subj}_MAIN_logfile_7.csv")
    vr_path = os.path.join(data_dir, subj, f"{subj}_valuerecall.csv")

    if not os.path.exists(log_path):
        return {}

    log_df = pd.read_csv(log_path)
    for col in ("phase", "event"):
        if col in log_df.columns:
            log_df[col] = log_df[col].astype(str).str.lower()

    # Build true outcome map from encoding rows
    enc = log_df[(log_df["phase"] == "encoding") & (log_df["event"] == "image")]
    true_map: dict[tuple, float] = {}
    for _, row in enc.iterrows():
        game = int(row["game"])
        image = str(row["image"])
        try:
            true_map[(game, image)] = float(row["outcome"])
        except (TypeError, ValueError):
            true_map[(game, image)] = np.nan

    # Get value_recall item order from logfile
    vr_log = log_df[
        (log_df["phase"] == "memory") & (log_df["event"] == "value_recall")
    ]

    # Try to load recalled values
    recalled_map: dict[tuple, float] = {}
    if os.path.exists(vr_path):
        vr_df = pd.read_csv(vr_path)
        for game in sorted(vr_log["game"].unique()):
            game_int = int(game)
            game_items = vr_log[vr_log["game"] == game]["image"].tolist()
            recalled_values = _parse_valuerecall(vr_df, game_int)
            for idx, image in enumerate(game_items):
                if idx < len(recalled_values) and recalled_values[idx] is not None:
                    recalled_map[(game_int, str(image))] = float(
                        recalled_values[idx]
                    )

    # Build final mapping: recalled value if available, else true outcome
    mapping: dict[tuple, tuple[str, float]] = {}
    for key in true_map:
        if key in recalled_map:
            val = recalled_map[key]
        else:
            val = true_map[key]

        if not np.isfinite(val):
            valence = "neutral"
        elif val > 0:
            valence = "positive"
        elif val < 0:
            valence = "negative"
        else:
            valence = "neutral"
        mapping[key] = (valence, val)

    return mapping
