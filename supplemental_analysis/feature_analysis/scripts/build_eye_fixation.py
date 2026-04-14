"""Build the two eye-gaze fixation datasets used by compute_feature_stats.R.

Reads the fixation-proportion CSV produced by the main paper pipeline
(`output/eyegaze/stats/choice_fixation_relevant_trial_deltas_long_duration.csv`)
and joins `feature_dim` from `feature_analyses/data/choices.csv`
(so `build_choices.py` must run first).

Writes:
  - feature_analyses/data/eye_fixation_long.csv      (one row per trial
    per valence category; two rows per trial)
  - feature_analyses/data/eye_fixation_relevant.csv  (trial-level,
    valence collapsed; carries `delta_relevant`)
"""

from __future__ import annotations

import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(FEATURE_DIR, "..", ".."))

EYE_SRC = os.path.join(
    REPO_ROOT, "output", "eyegaze", "stats",
    "choice_fixation_relevant_trial_deltas_long_duration.csv",
)
CHOICES = os.path.join(FEATURE_DIR, "data", "choices.csv")
OUT_DIR = os.path.join(FEATURE_DIR, "data")


def main():
    if not os.path.exists(CHOICES):
        raise SystemExit(
            f"{CHOICES} not found. Run build_choices.py first."
        )

    os.makedirs(OUT_DIR, exist_ok=True)

    eye = pd.read_csv(EYE_SRC)
    ch = pd.read_csv(CHOICES)

    keys = ["subject", "game", "trial_number"]
    for df, name in [(eye, "eye"), (ch, "choices")]:
        missing = [k for k in keys if k not in df.columns]
        if missing:
            raise KeyError(f"{name} missing keys: {missing}")

    ch_small = (
        ch[keys + ["feature_dim", "option", "choice", "correct"]]
        .drop_duplicates(keys)
    )
    merged = eye.merge(ch_small, on=keys, how="left", validate="many_to_one")

    unmatched = merged["feature_dim"].isna().sum()
    if unmatched > 0:
        raise ValueError(
            f"{unmatched} eye-gaze rows failed to merge with choices.csv"
        )

    n_subj = merged["subject"].nunique()
    print(f"Eye-gaze N subjects: {n_subj} (expected 41; exclusions 131, 107)")
    assert n_subj == 41, f"expected 41 subjects, got {n_subj}"

    long_path = os.path.join(OUT_DIR, "eye_fixation_long.csv")
    merged.to_csv(long_path, index=False)
    print(f"Wrote {long_path} ({len(merged)} rows)")

    collapsed = (
        merged.groupby(
            ["subject", "game", "trial_number", "decision_label",
             "feature_dim", "option", "choice", "correct"],
            dropna=False,
        )
        .agg(
            prop_relevant=("prop", "sum"),
            chance_relevant=("chance", "sum"),
            delta_relevant=("delta_from_chance", "sum"),
        )
        .reset_index()
    )
    collapsed_path = os.path.join(OUT_DIR, "eye_fixation_relevant.csv")
    collapsed.to_csv(collapsed_path, index=False)
    print(f"Wrote {collapsed_path} ({len(collapsed)} rows)")


if __name__ == "__main__":
    main()
