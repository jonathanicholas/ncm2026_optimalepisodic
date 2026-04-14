"""Build the per-trial choices dataset used by compute_feature_stats.R.

Reads raw subject files from the top-level `data/` directory via helper
functions in `analysis/analyze_behavior.py`, augments each choice row
with `trial_position` (1..8 within block) and `feature_dim`
(animacy / environment / size / texture), and writes the result to
`feature_analyses/data/choices.csv`.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import pandas as pd

# Resolve repo root (two levels above this script's scripts/ dir).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(FEATURE_DIR, ".."))

sys.path.insert(0, os.path.join(REPO_ROOT, "analysis"))

from analyze_behavior import (  # noqa: E402
    list_subjects,
    load_main_logfile,
    build_subject_choice_dataset,
)

DATA_ROOT = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(FEATURE_DIR, "data")

OPTION_TO_FEATURE_DIM = {
    "Animal": "animacy",
    "Object": "animacy",
    "Land": "environment",
    "Sea": "environment",
    "Large": "size",
    "Small": "size",
    "Solid": "texture",
    "Pattern": "texture",
}


def extract_choice_trial_numbers(df_main: pd.DataFrame) -> pd.DataFrame:
    """Rank choice-phase trial_numbers within each game (1..8)."""
    ch = df_main[(df_main["phase"] == "choice") & (df_main["event"] == "choice")].copy()
    if "trial_number" not in ch.columns:
        raise KeyError("trial_number column missing from main logfile")
    ch["trial_number"] = pd.to_numeric(ch["trial_number"], errors="coerce")
    ch["game"] = pd.to_numeric(ch["game"], errors="coerce").astype("Int64")
    ch = ch.dropna(subset=["game", "option", "trial_number"]).copy()
    ch["game"] = ch["game"].astype(int)
    ch["trial_position"] = (
        ch.groupby("game")["trial_number"].rank(method="dense").astype(int)
    )
    return ch[["game", "option", "trial_number", "trial_position"]]


def build_subject(subid: str) -> Optional[pd.DataFrame]:
    df_choices = build_subject_choice_dataset(subid, DATA_ROOT)
    if df_choices is None or len(df_choices) == 0:
        return None
    df_main = load_main_logfile(subid, DATA_ROOT)
    if df_main is None:
        return None
    trials = extract_choice_trial_numbers(df_main)
    merged = df_choices.merge(trials, on=["game", "option"], how="left",
                              validate="one_to_one")
    merged["feature_dim"] = merged["option"].map(OPTION_TO_FEATURE_DIM)
    return merged


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for sid in list_subjects(DATA_ROOT):
        df = build_subject(sid)
        if df is None or len(df) == 0:
            continue
        frames.append(df)
    if not frames:
        raise SystemExit("No subject data found.")
    df_all = pd.concat(frames, ignore_index=True)

    n_subj = df_all["subject"].nunique()
    if n_subj != 43:
        print(f"WARNING: expected 43 subjects, got {n_subj}", file=sys.stderr)

    missing_fd = df_all["feature_dim"].isna().sum()
    if missing_fd > 0:
        bad = df_all.loc[df_all["feature_dim"].isna(), "option"].unique()
        raise ValueError(f"feature_dim NA for options: {bad}")

    pos_range = df_all["trial_position"].dropna().astype(int)
    if not set(pos_range.unique()).issubset(set(range(1, 9))):
        raise ValueError(
            f"trial_position values outside 1..8: {sorted(pos_range.unique())}"
        )

    out_path = os.path.join(OUT_DIR, "choices.csv")
    df_all.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(f"N subjects: {n_subj}, N rows: {len(df_all)}")
    print("feature_dim counts:")
    print(df_all["feature_dim"].value_counts().to_string())


if __name__ == "__main__":
    main()
