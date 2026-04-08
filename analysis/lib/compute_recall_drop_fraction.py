#!/usr/bin/env python3
"""Compute the average proportion of fixation time during significant recall timepoints.

Reads the group time course CSV (with cluster-permutation `significant` column)
and per-subject fixation time course CSVs, then computes for each subject the
mean proportion_duration at significant timepoints.  The group mean of these
per-subject values is saved as ``recall_sig_timepoint_prop_fix_time.csv``.

The complement (1 - mean) serves as the ``--drop-frac`` value for
``predict_choice_from_item_prop_time_interactions.py`` in the "droprecall"
analysis.

Usage
-----
    conda run -n analysis python analysis/lib/compute_recall_drop_fraction.py \
        --data-dir data \
        --group-csv output/eyegaze/recall/group_time_course_original_buffer_50.csv \
        --out-csv output/eyegaze/recall/recall_sig_timepoint_prop_fix_time.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EYETRACK_EXCLUDE_SUBJECTS = {"107", "131"}


def compute_recall_drop_fraction(
    data_dir: str,
    group_csv: str,
    roi_type: str = "original",
    buffer_size: int = 50,
) -> tuple[float, float, int]:
    """Return (mean_prop_fix_time_sig, sem, n_timepoints).

    Parameters
    ----------
    data_dir : str
        Root data directory containing per-subject folders.
    group_csv : str
        Path to group_time_course CSV with ``significant`` column.
    roi_type, buffer_size : str, int
        Used to construct per-subject filename.
    """
    # Identify significant timepoints from group time course.
    gtc = pd.read_csv(group_csv)
    sig_mask = gtc["significant"].astype(bool)
    sig_timepoints = set(gtc.loc[sig_mask, "time_point"].values)
    n_sig = int(sig_mask.sum())

    if n_sig == 0:
        raise ValueError("No significant timepoints found in group time course.")

    # Collect per-subject mean proportion_duration at significant timepoints.
    subject_means: list[float] = []
    subject_dirs = sorted(
        d
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d not in EYETRACK_EXCLUDE_SUBJECTS
    )

    for subid in subject_dirs:
        tc_file = os.path.join(
            data_dir, subid,
            f"{subid}_fixation_time_course_{roi_type}_buffer_{buffer_size}.csv",
        )
        if not os.path.isfile(tc_file):
            continue
        tc = pd.read_csv(tc_file)
        sig_rows = tc[tc["time_point"].isin(sig_timepoints)]
        if sig_rows.empty:
            continue
        subj_mean = float(np.nanmean(sig_rows["proportion_duration"].values))
        subject_means.append(subj_mean)

    if not subject_means:
        raise ValueError("No subject data found for computing recall drop fraction.")

    arr = np.array(subject_means)
    mean_val = float(np.mean(arr))
    sem_val = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
    return mean_val, sem_val, n_sig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute average proportion of fixation time during significant recall timepoints.",
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Root data directory containing per-subject folders.",
    )
    parser.add_argument(
        "--group-csv",
        default="output/eyegaze/recall/group_time_course_original_buffer_50.csv",
        help="Group time course CSV with 'significant' column.",
    )
    parser.add_argument(
        "--out-csv",
        default="output/eyegaze/recall/recall_sig_timepoint_prop_fix_time.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--roi-type", default="original")
    parser.add_argument("--buffer-size", type=int, default=50)
    args = parser.parse_args()

    mean_val, sem_val, n_sig = compute_recall_drop_fraction(
        data_dir=args.data_dir,
        group_csv=args.group_csv,
        roi_type=args.roi_type,
        buffer_size=args.buffer_size,
    )

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    pd.DataFrame([{
        "mean_prop_fix_time_sig": mean_val,
        "sem": sem_val,
        "n_timepoints": n_sig,
    }]).to_csv(args.out_csv, index=False)

    drop_frac = 1.0 - mean_val
    print(f"mean_prop_fix_time_sig = {mean_val:.6f}")
    print(f"sem = {sem_val:.6f}")
    print(f"n_significant_timepoints = {n_sig}")
    print(f"drop_frac (1 - mean) = {drop_frac:.6f}")
    print(f"Saved to {args.out_csv}")


if __name__ == "__main__":
    main()
