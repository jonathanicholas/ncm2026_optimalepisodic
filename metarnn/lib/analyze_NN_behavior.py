#!/usr/bin/env python3
"""Compile NN-simulation behavioral data into trial-level and subject-level CSVs.

NN counterpart to analysis/analyze_behavior.py. Uses true offer values and
total fixation time (no memory phase or meaningful RT in NN simulations).

Inputs:  <root>/data/<SUBJ>/<SUBJ>_MAIN_logfile_7.csv and fixation CSVs.
Outputs: nn_trial_level_behavior.csv, subject_behavior_summary.csv.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_subjects(data_root: str) -> List[str]:
    subs: List[str] = []
    if not os.path.exists(data_root):
        return subs
    for d in os.listdir(data_root):
        if not os.path.isdir(os.path.join(data_root, d)):
            continue
        if re.fullmatch(r"\d+", d):
            subs.append(d)
    subs.sort()
    return subs


def load_main_logfile(subid: str, data_root: str) -> Optional[pd.DataFrame]:
    path = os.path.join(data_root, subid, f"{subid}_MAIN_logfile_7.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)

    # Normalize types
    for col in ["game", "outcome", "choice", "correct", "rt", "true_value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize strings
    for col in ["phase", "event", "image", "option"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Some human logs have a BOM in the column name; handle defensively.
    if "\ufeffphase" in df.columns and "phase" not in df.columns:
        df = df.rename(columns={"\ufeffphase": "phase"})

    return df


def load_fixations(subid: str, root: str) -> Optional[pd.DataFrame]:
    # Prefer buffer_50 naming (matches human pipeline), fallback to original.
    p1 = os.path.join(root, "data", subid, f"{subid}_fixations_df_original_buffer_50.csv")
    p2 = os.path.join(root, "data", subid, f"{subid}_fixations_df_original.csv")
    path = p1 if os.path.exists(p1) else p2
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)

    for col in ["phase", "event", "roi_content", "option"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    for col in ["game", "trial_number", "fix_duration_bounded", "fix_duration_full", "choice", "correct", "rt", "true_value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def extract_game_items_from_encoding_images(df_main: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    enc = df_main[(df_main["phase"] == "encoding") & (df_main["event"] == "image")].copy()
    enc = enc[["game", "image", "outcome"]].dropna(subset=["game", "image"]).copy()
    game_items: Dict[int, pd.DataFrame] = {}
    for g, gdf in enc.groupby("game"):
        gdf = gdf.drop_duplicates(subset=["image"]).copy()
        game_items[int(g)] = gdf.reset_index(drop=True)
    return game_items


def compute_true_offer_value(option: str, game_items_df: pd.DataFrame) -> float:
    token = str(option).strip()
    mask = game_items_df["image"].astype(str).str.split("_").apply(lambda parts: token in parts)
    return float(np.nansum(game_items_df.loc[mask, "outcome"].astype(float)))


def build_subject_trial_dataset(subid: str, root: str) -> Optional[pd.DataFrame]:
    df_main = load_main_logfile(subid, os.path.join(root, "data"))
    if df_main is None or len(df_main) == 0:
        return None

    df_fix = load_fixations(subid, root)
    if df_fix is None or len(df_fix) == 0:
        return None

    game_items = extract_game_items_from_encoding_images(df_main)

    # Choice trials (one per game for the NN compilation)
    choices = df_main[(df_main["phase"] == "choice") & (df_main["event"] == "choice")].copy()
    if len(choices) == 0:
        return None

    # Normalize choice: 1->1 (take), 2->0 (leave)
    choices["choice_bin"] = choices["choice"].replace({2: 0, 1: 1}).astype(float)

    # Fixation summary per (game, trial_number)
    choice_fix = df_fix[(df_fix["phase"] == "choice") & (df_fix["event"] == "choice")].copy()
    if len(choice_fix) == 0:
        return None

    dur_col = "fix_duration_bounded" if "fix_duration_bounded" in choice_fix.columns else "fix_duration_full"

    group_cols = ["game", "trial_number"]

    # Total fixation time per trial (all choice fixation rows)
    total_fix_time = (
        choice_fix.groupby(group_cols, sort=True)[dur_col]
        .sum(min_count=1)
        .rename("total_fix_time")
        .reset_index()
    )

    # Only count actual item ROIs (exclude none/fixation)
    item_mask = (
        choice_fix["roi_content"].notna()
        & (~choice_fix["roi_content"].isin(["none", "fixation"]))
        & (choice_fix["roi_content"].astype(str).str.contains("_"))
    )
    choice_fix_items = choice_fix.loc[item_mask].copy()

    unique_items = (
        choice_fix_items.groupby(group_cols, sort=True)["roi_content"]
        .nunique()
        .rename("unique_items_fixated")
        .reset_index()
    )

    # Relevant unique items: option token contained in item name
    if len(choice_fix_items) > 0 and "option" in choice_fix_items.columns:
        opts = choice_fix_items["option"].astype(str).fillna("")
        rois = choice_fix_items["roi_content"].astype(str).fillna("")
        rel_flags = [
            (opt != "") and (opt in roi.split("_") or opt in roi)
            for opt, roi in zip(opts.tolist(), rois.tolist())
        ]
        choice_fix_items["_is_relevant"] = rel_flags
        relevant_items = (
            choice_fix_items.loc[choice_fix_items["_is_relevant"]]
            .groupby(group_cols, sort=True)["roi_content"]
            .nunique()
            .rename("relevant_items_fixated")
            .reset_index()
        )
    else:
        relevant_items = pd.DataFrame(columns=group_cols + ["relevant_items_fixated"])

    fix_summ = total_fix_time.merge(unique_items, on=group_cols, how="left").merge(
        relevant_items, on=group_cols, how="left"
    )
    fix_summ["unique_items_fixated"] = fix_summ["unique_items_fixated"].fillna(0).astype(int)
    fix_summ["relevant_items_fixated"] = fix_summ["relevant_items_fixated"].fillna(0).astype(int)

    # Merge choice trials with fixation summaries by game (and trial_number if present)
    out_rows: List[dict] = []
    for _, row in choices.iterrows():
        game = int(row["game"]) if not pd.isna(row.get("game")) else None
        option = row.get("option", None)
        if game is None or option is None or pd.isna(option):
            continue
        items_df = game_items.get(game)
        if items_df is None:
            continue

        true_val = compute_true_offer_value(str(option), items_df)

        # There is exactly one choice trial per game in the compiled NN data
        # but keep a defensive lookup by (game, trial_number=1) and fallback to any.
        trial_num = int(row["trial_number"]) if "trial_number" in row and not pd.isna(row["trial_number"]) else 1
        match = fix_summ[(fix_summ["game"] == game) & (fix_summ["trial_number"] == trial_num)]
        if len(match) == 0:
            match = fix_summ[(fix_summ["game"] == game)]
        if len(match) == 0:
            continue
        ms = match.iloc[0]

        out_rows.append(
            {
                "subject": subid,
                "game": game,
                "trial_number": trial_num,
                "option": str(option),
                "choice": float(row["choice_bin"]) if not pd.isna(row.get("choice_bin")) else np.nan,
                "correct": float(row["correct"]) if not pd.isna(row.get("correct")) else np.nan,
                "true_offer_value": float(true_val),
                "total_fix_time": float(ms["total_fix_time"]),
                "unique_items_fixated": int(ms["unique_items_fixated"]),
                "relevant_items_fixated": int(ms.get("relevant_items_fixated", 0)),
            }
        )

    if len(out_rows) == 0:
        return None
    return pd.DataFrame(out_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile NN behavioral data into trial-level and subject-level CSVs.")
    parser.add_argument(
        "--root",
        default="metarnn/simulations/human_like",
        help="Root folder containing data/ subfolder for NN simulations.",
    )
    parser.add_argument(
        "--out_dir",
        default="output/behavior/stats",
        help="Directory to write output CSVs into.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional additional path to save the compiled trial-level dataframe as CSV.",
    )

    args = parser.parse_args()

    root = args.root
    data_root = os.path.join(root, "data")

    subjects = list_subjects(data_root)
    if len(subjects) == 0:
        raise RuntimeError(f"No subjects found under {data_root}")

    all_rows: List[pd.DataFrame] = []
    for sid in subjects:
        df = build_subject_trial_dataset(sid, root)
        if df is None or len(df) == 0:
            continue
        all_rows.append(df)

    if len(all_rows) == 0:
        raise RuntimeError("No NN trial-level datasets could be built.")

    df_all = pd.concat(all_rows, ignore_index=True)

    ensure_output_dir(args.out_dir)

    # Always save trial-level data
    trial_csv_path = os.path.join(args.out_dir, "nn_trial_level_behavior.csv")
    df_all.to_csv(trial_csv_path, index=False)
    print(f"Saved trial-level data ({len(df_all)} rows) to {trial_csv_path}")

    # Save subject-level behavior summary
    subj_summary = (
        df_all.groupby("subject")
        .agg(
            n_trials=("correct", "count"),
            accuracy=("correct", "mean"),
            mean_true_offer_value=("true_offer_value", "mean"),
            mean_total_fix_time=("total_fix_time", "mean"),
            mean_unique_items_fixated=("unique_items_fixated", "mean"),
            mean_relevant_items_fixated=("relevant_items_fixated", "mean"),
        )
        .reset_index()
    )
    subj_csv_path = os.path.join(args.out_dir, "subject_behavior_summary.csv")
    subj_summary.to_csv(subj_csv_path, index=False)
    print(f"Saved subject-level summary ({len(subj_summary)} subjects) to {subj_csv_path}")

    # Optional additional CSV output
    if args.csv is not None:
        ensure_output_dir(os.path.dirname(args.csv) or ".")
        df_all.to_csv(args.csv, index=False)
        print(f"Saved additional trial-level CSV to {args.csv}")


if __name__ == "__main__":
    main()
