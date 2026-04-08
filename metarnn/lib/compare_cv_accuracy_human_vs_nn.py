#!/usr/bin/env python3
"""Compare human vs NN (with prop-recall drop) CV accuracy.

Reads both summary CSVs, computes the human 95% CI, checks whether the
NN drop-recall CV accuracy falls within it, and saves the result.

Usage
-----
    python metarnn/lib/compare_cv_accuracy_human_vs_nn.py \
        --nn-drop-csv <path> --human-csv <path> --out-dir <path>
"""

from __future__ import annotations

import argparse
import os

import pandas as pd


def _read_cv(path: str) -> tuple[float, float]:
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
        description="Compare human vs NN (prop-recall drop) CV accuracy."
    )
    parser.add_argument("--nn-drop-csv", required=True,
                        help="NN drop-recall CV summary CSV.")
    parser.add_argument("--human-csv", required=True,
                        help="Human CV accuracy summary CSV.")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory.")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    h_mean, h_sem = _read_cv(args.human_csv)
    nn_mean, nn_sem = _read_cv(args.nn_drop_csv)

    h_ci_lo = h_mean - 1.96 * h_sem
    h_ci_hi = h_mean + 1.96 * h_sem
    within_ci = h_ci_lo <= nn_mean <= h_ci_hi

    result = pd.DataFrame([{
        "human_mean": h_mean,
        "human_sem": h_sem,
        "human_ci_lo": h_ci_lo,
        "human_ci_hi": h_ci_hi,
        "nn_drop_mean": nn_mean,
        "nn_drop_sem": nn_sem,
        "within_ci": within_ci,
    }])

    out_path = os.path.join(out_dir, "cv_accuracy_human_vs_nn_drop_comparison.csv")
    result.to_csv(out_path, index=False)

    print(f"Human CV: {h_mean:.4f} ± {h_sem:.4f} (95% CI [{h_ci_lo:.4f}, {h_ci_hi:.4f}])")
    print(f"NN drop CV: {nn_mean:.4f} ± {nn_sem:.4f}")
    print(f"NN within human 95% CI: {within_ci}")
    print(f"Result saved to {out_path}")


if __name__ == "__main__":
    main()
