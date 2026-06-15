#!/usr/bin/env python3
"""Plot CV accuracy as a function of the proportion of fixations retained.

Reads a consolidated drop-sweep CSV (with columns drop_pct, retained_frac,
cv_mean, cv_sem) and overlays human CV accuracy and recall-proportion
reference lines.

Usage
-----
    conda run -n analysis python metarnn/lib/plot_prop_drop_supplement.py \
        --sweep-csv <path_to_drop_sweep_summary.csv> \
        --human-pred-csv <path_to_human_summary.csv> \
        --recall-csv <path_to_recall_sig_timepoint_prop_fix_time.csv> \
        --out-dir <output_directory> \
        --tag <simulation_tag>
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _read_cv_from_summary(path: str) -> tuple[float, float]:
    """Return (cv_mean, cv_sem) from a prediction summary CSV."""
    df = pd.read_csv(path)
    if "feature_set" in df.columns:
        df = df[df["feature_set"].astype(str) == "location_interactions"]
    if "visit_type" in df.columns:
        df = df[df["visit_type"].astype(str) == "all"]
    row = df.iloc[0]
    return float(row["cv_mean"]), float(row["cv_sem"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot CV accuracy vs proportion of fixations retained."
    )
    parser.add_argument(
        "--sweep-csv", required=True,
        help="Consolidated drop-sweep CSV (drop_pct, retained_frac, cv_mean, cv_sem).",
    )
    parser.add_argument(
        "--human-pred-csv", required=True,
        help="Human CV accuracy summary CSV.",
    )
    parser.add_argument(
        "--recall-csv", required=True,
        help="Recall-proportion CSV (mean_prop_fix_time_sig, sem).",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--tag", required=True, help="Simulation tag.")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # --- Load drop-sweep data ---
    sweep = pd.read_csv(args.sweep_csv)
    sweep = sweep.sort_values("retained_frac").reset_index(drop=True)
    x = sweep["retained_frac"].values
    y = sweep["cv_mean"].values
    ci = 1.96 * sweep["cv_sem"].values

    # --- Load human reference ---
    h_mean, h_sem = _read_cv_from_summary(args.human_pred_csv)
    h_ci = 1.96 * h_sem

    # --- Load recall proportion reference ---
    rdf = pd.read_csv(args.recall_csv)
    r_mean = float(rdf["mean_prop_fix_time_sig"].iloc[0])
    r_sem = float(rdf["sem"].iloc[0])
    r_lo = r_mean - 1.96 * r_sem
    r_hi = r_mean + 1.96 * r_sem

    # --- Plot ---
    sns.set_context("poster")
    with plt.rc_context(
        {
            "font.family": "Arial",
            "axes.titlesize": 24,
            "axes.labelsize": 28,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
        }
    ):
        fig, ax = plt.subplots(figsize=(8, 7))

        # NN drop-fraction curve
        ax.plot(x, y, "o-", color="black", markersize=8, linewidth=2, zorder=3)
        ax.fill_between(x, y - ci, y + ci, color="black", alpha=0.15,
                         edgecolor="none", zorder=2)
        ax.text(
            x[-1], y[-1] + ci[-1] + 0.003, "RNN",
            fontsize=20, color="black", va="bottom", ha="right",
        )

        # Human CV accuracy band
        ax.axhline(h_mean, color="gray", linestyle="-", linewidth=2.5, zorder=1)
        ax.axhspan(
            h_mean - h_ci, h_mean + h_ci,
            facecolor="gray", alpha=0.2, edgecolor="none", zorder=0,
        )
        ax.text(
            0.06, h_mean + h_ci + 0.003, "Human",
            fontsize=20, color="gray", va="bottom", ha="left",
        )

        # Recall proportion vertical reference
        ax.axvline(r_mean, color="gray", linestyle="-", linewidth=2.5, zorder=1)
        ax.axvspan(r_lo, r_hi, facecolor="gray", alpha=0.2, edgecolor="none", zorder=0)

        # Axis formatting (set limits before placing recall text)
        ax.set_xlabel("Proportion of Fixations Retained")
        ax.set_ylabel("Choice Prediction Accuracy")
        ax.set_xticks(np.arange(0.1, 1.05, 0.2))
        ax.set_xlim(0.05, 1.05)
        ax.set_ylim(0.5, 0.75)

        # Place recall text after y-limits are set
        ax.text(
            r_hi + 0.01, ax.get_ylim()[1], "Human Recall Proportion",
            va="top", ha="left", fontsize=20, color="gray",
        )

        # Chance line
        ax.axhline(0.5, color="black", linestyle=":", linewidth=1.5, alpha=0.5, zorder=1)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"propDropSupplement_{args.tag}.pdf")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved prop-drop supplement to {out_path}")

    # --- Write metadata ---
    meta_path = os.path.join(out_dir, f"propDropSupplement_{args.tag}.meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"sweep_csv: {os.path.abspath(args.sweep_csv)}\n")
        f.write(f"human_pred_csv: {os.path.abspath(args.human_pred_csv)}\n")
        f.write(f"recall_csv: {os.path.abspath(args.recall_csv)}\n")
        f.write(f"out_dir: {out_dir}\n")
        f.write(f"tag: {args.tag}\n")


if __name__ == "__main__":
    main()
