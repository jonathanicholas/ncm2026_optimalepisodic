"""Plot fold-mean parameter bars (theta, d, sigma) with SEM.

Style matches the held-out log-likelihood bar panel:
- single gray bar with black outline
- black SEM marker/error line at the mean
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _sem(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size <= 1:
        return 0.0
    return float(np.std(v, ddof=1) / np.sqrt(v.size))


def _draw_bar_panel(ax: plt.Axes, *, mean_val: float, sem_val: float, y_label: str) -> None:
    ax.bar(
        [0],
        [mean_val],
        width=0.55,
        color="0.5",
        edgecolor="black",
        linewidth=2,
        zorder=1,
    )
    ax.errorbar(
        [0],
        [mean_val],
        yerr=[sem_val],
        fmt="_",
        markersize=16,
        color="black",
        linewidth=1.3,
        capsize=0,
        zorder=3,
    )
    ax.axhline(0.0, linestyle="--", color="0.25", linewidth=1)
    ax.set_xlim(-0.8, 0.8)
    ax.set_xticks([])
    ax.set_ylabel(y_label)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def _plot_multi_panel(
    summary_rows: List[dict],
    *,
    out_png: Path,
    out_pdf: Path,
    panel_width: float,
    panel_height: float,
) -> None:
    if len(summary_rows) == 0:
        raise ValueError("No parameter summaries available for plotting")

    n_panels = len(summary_rows)
    sns.set_context("poster")
    sns.set_style("ticks")
    with plt.rc_context(
        {
            "font.family": "Arial",
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 14,
            "ytick.labelsize": 18,
            "lines.solid_capstyle": "butt",
            "lines.dash_capstyle": "butt",
        }
    ):
        fig, axes = plt.subplots(1, n_panels, figsize=(panel_width * n_panels, panel_height))
        if n_panels == 1:
            axes = [axes]
        for ax, row in zip(axes, summary_rows):
            _draw_bar_panel(
                ax,
                mean_val=float(row["mean"]),
                sem_val=float(row["sem"]),
                y_label=str(row["parameter"]),
            )
        fig.tight_layout()
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot theta/d/sigma mean±SEM across k-fold fits.")
    parser.add_argument(
        "--kfold-summary-csv",
        type=str,
        default="output/addm/kfold/rtTrans_recalled_final/addm_kfold_fit_summary_free3-rtTrans.csv",
        help="Fold-level kfold fit summary CSV (typically free3 model)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="output/addm/kfold_compare/rtTrans_recalled_final",
        help="Directory to save output plots",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag appended to output filenames",
    )
    parser.add_argument(
        "--mode-filter",
        type=str,
        default="fit",
        help="Optional row filter on mode column (default: fit)",
    )
    parser.add_argument(
        "--panel-width",
        type=float,
        default=3.6,
        help="Width (inches) of each panel in the multi-panel figure (default: 3.6)",
    )
    parser.add_argument(
        "--panel-height",
        type=float,
        default=4.2,
        help="Height (inches) of the multi-panel figure (default: 4.2)",
    )
    args = parser.parse_args()

    in_csv = Path(args.kfold_summary_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_csv.exists():
        raise FileNotFoundError(f"Missing kfold summary CSV: {in_csv}")

    df = pd.read_csv(in_csv)
    if "mode" in df.columns and str(args.mode_filter).strip() != "":
        df = df[df["mode"].astype(str) == str(args.mode_filter)].copy()

    params: List[str] = ["theta", "d", "sigma"]
    for p in params:
        if p not in df.columns:
            raise ValueError(f"Column '{p}' not found in {in_csv}")

    summary_rows = []
    stem_base = "kfold_param"
    if args.tag:
        stem_base = f"{stem_base}_{args.tag}"

    for p in params:
        vals = pd.to_numeric(df[p], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        mean_val = float(np.mean(vals))
        sem_val = _sem(vals)
        summary_rows.append(
            {
                "parameter": p,
                "n_folds": int(vals.size),
                "mean": mean_val,
                "sem": sem_val,
            }
        )

    if len(summary_rows) > 0:
        out_png = out_dir / f"{stem_base}_all_params_bar_sem.png"
        out_pdf = out_dir / f"{stem_base}_all_params_bar_sem.pdf"
        _plot_multi_panel(
            summary_rows,
            out_png=out_png,
            out_pdf=out_pdf,
            panel_width=float(args.panel_width),
            panel_height=float(args.panel_height),
        )
        print(f"Wrote: {out_png}")
        print(f"Wrote: {out_pdf}")

        out_csv = out_dir / f"{stem_base}_summary.csv"
        pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
        print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
