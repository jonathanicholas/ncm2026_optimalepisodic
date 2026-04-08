"""Plot fixation advantage as half-violin distributions with summary statistics.

Computes fixation advantage (cumulative fixation time to the currently fixated
item minus the mean cumulative fixation time to the other items) and produces a
single-panel horizontal half-violin figure for All / Relevant / Irrelevant items.

Also saves a CSV table with summary statistics including the proportion of
fixation advantages below zero.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    base_dir: Path
    figures_dir: Path
    excluded_subjects: Tuple[str, ...] = ("107", "131")
    time_unit_label: str = "ms"
    output_tag: str = ""
    show_plots: bool = True
    other_items_mode: str = "all"
    advantage_timepoint: str = "pre"
    exclude_outliers: bool = False
    outlier_sd: float = 3.0


# ---------------------------------------------------------------------------
# Helpers (copied from compute_fixation_advantage.py for self-containment)
# ---------------------------------------------------------------------------

def _exclude_outliers_sd(series: pd.Series, *, k: float) -> pd.Series:
    """Return a copy where values outside mean +/- k*SD are set to NaN."""
    s = pd.to_numeric(series, errors="coerce")
    v = s[np.isfinite(s)]
    if v.size < 2:
        return s
    mu = float(v.mean())
    sd = float(v.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return s
    lo = mu - float(k) * sd
    hi = mu + float(k) * sd
    out = s.copy()
    mask = np.isfinite(out) & ((out < lo) | (out > hi))
    out.loc[mask] = np.nan
    return out


def _tagged_filename(name: str, *, tag: str, ext: str) -> str:
    tag = str(tag).strip()
    suffix = f"_{tag}" if tag else ""
    return f"{name}{suffix}.{ext.lstrip('.')}"


def _list_numeric_subdirs(parent: Path) -> List[str]:
    out: List[str] = []
    if not parent.exists():
        return out
    for p in sorted(parent.iterdir()):
        if p.is_dir() and p.name.isdigit():
            out.append(p.name)
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_fixations(
    output_dir: Path, excluded_subjects: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """Load fixation data for all subjects."""
    excluded_subjects_set = set(str(s) for s in (excluded_subjects or ["107", "131"]))

    all_dfs = []
    for subid in _list_numeric_subdirs(output_dir):
        if subid in excluded_subjects_set:
            print(f"Excluding subject {subid}")
            continue

        sub_dir = output_dir / subid
        fix_file = sub_dir / f"{subid}_fixations_for_modeling.csv"
        if not fix_file.exists():
            print(f"Warning: Fixation file not found for subject {subid}")
            continue

        df = pd.read_csv(fix_file)
        all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError(f"No fixation files found under {output_dir}")

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"Loaded fixations for {len(all_dfs)} subjects")
    print(f"Total fixations: {len(combined)}")

    required = [
        "subject",
        "game",
        "trial_number",
        "item_index",
        "is_relevant",
        "fix_start",
        "fix_duration_bounded",
    ]
    missing = [c for c in required if c not in combined.columns]
    if missing:
        raise ValueError(f"Fixation data missing required columns: {missing}")

    combined["subject"] = combined["subject"].astype(str)
    combined["game"] = pd.to_numeric(combined["game"], errors="coerce")
    combined["trial_number"] = pd.to_numeric(combined["trial_number"], errors="coerce")
    combined["item_index"] = pd.to_numeric(combined["item_index"], errors="coerce")
    combined["is_relevant"] = pd.to_numeric(combined["is_relevant"], errors="coerce")
    combined["fix_start"] = pd.to_numeric(combined["fix_start"], errors="coerce")
    combined["fix_duration_bounded"] = pd.to_numeric(
        combined["fix_duration_bounded"], errors="coerce"
    )
    combined = combined.dropna(
        subset=[
            "subject", "game", "trial_number", "item_index",
            "is_relevant", "fix_start", "fix_duration_bounded",
        ]
    ).copy()
    combined["game"] = combined["game"].astype(int)
    combined["trial_number"] = combined["trial_number"].astype(int)
    combined["item_index"] = combined["item_index"].astype(int)
    combined["is_relevant"] = combined["is_relevant"].astype(int)

    return combined


# ---------------------------------------------------------------------------
# Fixation advantage computation
# ---------------------------------------------------------------------------

def compute_fixation_advantages(
    fixations: pd.DataFrame,
    *,
    other_items_mode: str = "all",
    advantage_timepoint: str = "pre",
) -> pd.DataFrame:
    """Compute fixation advantage for each fixation (all/relevant/irrelevant)."""
    df = (
        fixations
        .sort_values(["subject", "game", "trial_number", "fix_start"])
        .reset_index(drop=True)
        .copy()
    )

    df["is_first_fixation_in_trial"] = (
        df.groupby(["subject", "game", "trial_number"]).cumcount() == 0
    )
    df["is_first_fixation_to_item"] = (
        df.groupby(["subject", "game", "trial_number", "item_index"]).cumcount() == 0
    )
    df["cumulative_time"] = df.groupby(
        ["subject", "game", "trial_number", "item_index"]
    )["fix_duration_bounded"].cumsum()
    df["cumulative_time_pre"] = (
        df["cumulative_time"] - df["fix_duration_bounded"]
    ).clip(lower=0)

    other_items_mode = str(other_items_mode).strip().lower()
    if other_items_mode not in {"all", "seen"}:
        raise ValueError("other_items_mode must be one of {'all','seen'}")

    advantage_timepoint = str(advantage_timepoint).strip().lower()
    if advantage_timepoint not in {"pre", "post"}:
        raise ValueError("advantage_timepoint must be one of {'pre','post'}")

    adv_all = np.empty(len(df), dtype=float)
    adv_rel = np.full(len(df), np.nan, dtype=float)
    adv_irr = np.full(len(df), np.nan, dtype=float)

    n_items_all = 6
    n_items_rel = 3
    n_items_irr = 3

    for _, idx in df.groupby(["subject", "game", "trial_number"]).groups.items():
        items = df.loc[idx, "item_index"].to_numpy(dtype=int)
        durs = df.loc[idx, "fix_duration_bounded"].to_numpy(dtype=float)
        rels = df.loc[idx, "is_relevant"].to_numpy(dtype=int)

        cum_all: Dict[int, float] = {}
        sum_all = 0.0
        cum_rel: Dict[int, float] = {}
        sum_rel = 0.0
        cum_irr: Dict[int, float] = {}
        sum_irr = 0.0

        for k, row_pos in enumerate(idx):
            item = int(items[k])
            dur = float(durs[k])

            # ---- all items ----
            prev = cum_all.get(item, 0.0)
            item_seen_pre = item in cum_all
            sum_all_pre = sum_all
            n_seen_all_pre = len(cum_all)

            if other_items_mode == "all":
                mean_other_all_pre = (sum_all_pre - prev) / (n_items_all - 1)
            else:
                if item_seen_pre:
                    n_other = n_seen_all_pre - 1
                    sum_other = sum_all_pre - prev
                else:
                    n_other = n_seen_all_pre
                    sum_other = sum_all_pre
                mean_other_all_pre = (sum_other / n_other) if n_other > 0 else 0.0

            adv_pre_all = prev - mean_other_all_pre
            new = prev + dur
            cum_all[item] = new
            sum_all += dur

            if advantage_timepoint == "pre":
                adv_all[row_pos] = adv_pre_all
            else:
                if other_items_mode == "all":
                    mean_other_all_post = (sum_all - new) / (n_items_all - 1)
                else:
                    n_all_post = len(cum_all)
                    mean_other_all_post = (
                        (sum_all - new) / (n_all_post - 1) if n_all_post > 1 else 0.0
                    )
                adv_all[row_pos] = new - mean_other_all_post

            # ---- relevant / irrelevant ----
            if int(rels[k]) == 1:
                prev_r = cum_rel.get(item, 0.0)
                item_seen_pre_r = item in cum_rel
                sum_rel_pre = sum_rel
                n_seen_rel_pre = len(cum_rel)

                if other_items_mode == "all":
                    mean_other_rel_pre = (sum_rel_pre - prev_r) / (n_items_rel - 1)
                else:
                    if item_seen_pre_r:
                        n_other = n_seen_rel_pre - 1
                        sum_other = sum_rel_pre - prev_r
                    else:
                        n_other = n_seen_rel_pre
                        sum_other = sum_rel_pre
                    mean_other_rel_pre = (sum_other / n_other) if n_other > 0 else 0.0

                adv_pre_rel = prev_r - mean_other_rel_pre
                new_r = prev_r + dur
                cum_rel[item] = new_r
                sum_rel += dur

                if advantage_timepoint == "pre":
                    adv_rel[row_pos] = adv_pre_rel
                else:
                    if other_items_mode == "all":
                        mean_other_rel_post = (sum_rel - new_r) / (n_items_rel - 1)
                    else:
                        n_rel_post = len(cum_rel)
                        mean_other_rel_post = (
                            (sum_rel - new_r) / (n_rel_post - 1)
                            if n_rel_post > 1
                            else 0.0
                        )
                    adv_rel[row_pos] = new_r - mean_other_rel_post
            else:
                prev_i = cum_irr.get(item, 0.0)
                item_seen_pre_i = item in cum_irr
                sum_irr_pre = sum_irr
                n_seen_irr_pre = len(cum_irr)

                if other_items_mode == "all":
                    mean_other_irr_pre = (sum_irr_pre - prev_i) / (n_items_irr - 1)
                else:
                    if item_seen_pre_i:
                        n_other = n_seen_irr_pre - 1
                        sum_other = sum_irr_pre - prev_i
                    else:
                        n_other = n_seen_irr_pre
                        sum_other = sum_irr_pre
                    mean_other_irr_pre = (sum_other / n_other) if n_other > 0 else 0.0

                adv_pre_irr = prev_i - mean_other_irr_pre
                new_i = prev_i + dur
                cum_irr[item] = new_i
                sum_irr += dur

                if advantage_timepoint == "pre":
                    adv_irr[row_pos] = adv_pre_irr
                else:
                    if other_items_mode == "all":
                        mean_other_irr_post = (sum_irr - new_i) / (n_items_irr - 1)
                    else:
                        n_irr_post = len(cum_irr)
                        mean_other_irr_post = (
                            (sum_irr - new_i) / (n_irr_post - 1)
                            if n_irr_post > 1
                            else 0.0
                        )
                    adv_irr[row_pos] = new_i - mean_other_irr_post

    df["fixation_advantage_all"] = adv_all
    df["fixation_advantage_relevant"] = adv_rel
    df["fixation_advantage_irrelevant"] = adv_irr

    if advantage_timepoint == "pre":
        first_trial_fix = df["is_first_fixation_in_trial"].to_numpy(dtype=bool)
        df.loc[first_trial_fix, "fixation_advantage_all"] = np.nan
        df.loc[first_trial_fix, "fixation_advantage_relevant"] = np.nan
        df.loc[first_trial_fix, "fixation_advantage_irrelevant"] = np.nan

    return df


def _subset_advantage(df: pd.DataFrame, item_subset: str) -> pd.DataFrame:
    if item_subset == "all":
        out = df.copy()
        out["fixation_advantage"] = out["fixation_advantage_all"]
        return out
    if item_subset == "relevant":
        out = df[df["is_relevant"] == 1].copy()
        out["fixation_advantage"] = out["fixation_advantage_relevant"]
        return out
    if item_subset == "irrelevant":
        out = df[df["is_relevant"] == 0].copy()
        out["fixation_advantage"] = out["fixation_advantage_irrelevant"]
        return out
    raise ValueError(f"Invalid item_subset: {item_subset}")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary_stats(
    df_all: pd.DataFrame,
    df_relevant: pd.DataFrame,
    df_irrelevant: pd.DataFrame,
) -> pd.DataFrame:
    """Compute summary statistics for each category and return as a DataFrame.

    The statistical test uses subject-level medians to respect the hierarchical
    structure of the data (fixations nested within subjects). A Wilcoxon
    signed-rank test is run on the per-subject median fixation advantages
    against 0.
    """
    rows = []
    for label, df in [
        ("All", df_all),
        ("Relevant", df_relevant),
        ("Irrelevant", df_irrelevant),
    ]:
        vals = pd.to_numeric(df["fixation_advantage"], errors="coerce")
        vals = vals[np.isfinite(vals)]
        n = len(vals)
        if n == 0:
            rows.append({"category": label, "N_fixations": 0})
            continue

        # Subject-level medians for hierarchical test
        sub_col = df.loc[vals.index, "subject"]
        subject_medians = vals.groupby(sub_col).median()
        n_subjects = len(subject_medians)

        if n_subjects >= 10:
            wilcoxon_res = stats.wilcoxon(subject_medians, alternative="two-sided")
            wilcoxon_stat = wilcoxon_res.statistic
            wilcoxon_p = wilcoxon_res.pvalue
        else:
            wilcoxon_stat = np.nan
            wilcoxon_p = np.nan

        rows.append({
            "category": label,
            "N_fixations": n,
            "N_subjects": n_subjects,
            "mean": vals.mean(),
            "median": vals.median(),
            "std": vals.std(ddof=1),
            "Q1": vals.quantile(0.25),
            "Q3": vals.quantile(0.75),
            "IQR": vals.quantile(0.75) - vals.quantile(0.25),
            "min": vals.min(),
            "max": vals.max(),
            "prop_below_zero": (vals < 0).mean(),
            "prop_above_zero": (vals > 0).mean(),
            "prop_equal_zero": (vals == 0).mean(),
            "subject_mean_of_medians": subject_medians.mean(),
            "subject_std_of_medians": subject_medians.std(ddof=1),
            "wilcoxon_stat": wilcoxon_stat,
            "wilcoxon_p": wilcoxon_p,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    "All": "#888888",
    "Relevant": "#6fc7eb",
    "Irrelevant": "#ba7caf",
}


def plot_half_violin(
    df_all: pd.DataFrame,
    df_relevant: pd.DataFrame,
    df_irrelevant: pd.DataFrame,
    output_dir: Path,
    *,
    time_unit_label: str = "ms",
    output_tag: str = "",
    show: bool = True,
) -> None:
    """Create a horizontal half-violin plot (density above) for fixation advantage."""

    # Build long-form data for seaborn
    records = []
    for label, df in [
        ("All", df_all),
        ("Relevant", df_relevant),
        ("Irrelevant", df_irrelevant),
    ]:
        vals = pd.to_numeric(df["fixation_advantage"], errors="coerce").dropna()
        sub = pd.DataFrame({"fixation_advantage": vals.values, "category": label})
        records.append(sub)
    plot_df = pd.concat(records, ignore_index=True)

    # Order: All at top, Irrelevant at bottom
    cat_order = ["All", "Relevant", "Irrelevant"]

    # Apply styling
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["xtick.major.width"] = 2
    plt.rcParams["ytick.major.width"] = 2

    with plt.rc_context({
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    }):
        fig, ax = plt.subplots(figsize=(10, 5))

        sns.violinplot(
            data=plot_df,
            x="fixation_advantage",
            y="category",
            hue="category",
            order=cat_order,
            hue_order=cat_order,
            orient="h",
            inner="quartile",
            palette=COLORS,
            saturation=1.0,
            linewidth=0,
            cut=0,
            bw_adjust=3,
            legend=False,
            ax=ax,
        )

        # Clip each violin body to show only the top half (density above baseline)
        for i, body in enumerate(ax.collections):
            paths = body.get_paths()
            for path in paths:
                verts = path.vertices
                center_y = i
                verts[:, 1] = np.clip(verts[:, 1], -np.inf, center_y)

        # Clip quartile/median lines to the upper half as well
        for line in ax.lines:
            ydata = line.get_ydata()
            if len(ydata) == 2:
                # Each quartile line spans the violin width vertically.
                # Determine which category it belongs to (nearest integer y).
                center_y = round(np.mean(ydata))
                # Clip the lower end to the center line
                new_y = np.clip(ydata, -np.inf, center_y)
                line.set_ydata(new_y)
            line.set_color("black")
            line.set_linewidth(1.2)

        # Reference line at 0
        ax.axvline(0, color="black", linestyle="--", linewidth=1.5, zorder=0)

        # Limit x-axis to the central 95% of the pooled data
        all_vals = plot_df["fixation_advantage"].dropna()
        lo, hi = float(all_vals.quantile(0.025)), float(all_vals.quantile(0.975))
        margin = (hi - lo) * 0.05
        ax.set_xlim(lo - margin, hi + margin)

        ax.set_xlabel(f"Fixation Advantage ({time_unit_label})")
        ax.set_ylabel("")
        ax.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()

        # Save
        out_pdf = output_dir / _tagged_filename(
            "fixation_advantage_violin", tag=output_tag, ext="pdf"
        )
        out_png = output_dir / _tagged_filename(
            "fixation_advantage_violin", tag=output_tag, ext="png"
        )
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.1)
        fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.1)
        print(f"Saved figure to {out_pdf}")
        print(f"Saved figure to {out_png}")

        if show:
            plt.show()
        else:
            plt.close(fig)


def plot_subject_medians(
    df_all: pd.DataFrame,
    df_relevant: pd.DataFrame,
    df_irrelevant: pd.DataFrame,
    output_dir: Path,
    *,
    time_unit_label: str = "ms",
    output_tag: str = "",
    show: bool = True,
) -> None:
    """Plot individual subject medians with group-level median and SEM."""

    cat_order = ["All", "Relevant", "Irrelevant"]

    # Compute per-subject medians for each category
    records = []
    for label, df in [
        ("All", df_all),
        ("Relevant", df_relevant),
        ("Irrelevant", df_irrelevant),
    ]:
        vals = pd.to_numeric(df["fixation_advantage"], errors="coerce")
        vals = vals[np.isfinite(vals)]
        sub_col = df.loc[vals.index, "subject"]
        sub_medians = vals.groupby(sub_col).median()
        for subj, med in sub_medians.items():
            records.append({
                "subject": subj,
                "category": label,
                "subject_median": med,
            })
    plot_df = pd.DataFrame(records)

    # Apply styling
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["xtick.major.width"] = 2
    plt.rcParams["ytick.major.width"] = 2

    with plt.rc_context({
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    }):
        fig, ax = plt.subplots(figsize=(10, 5))

        # Individual subject points (stripplot-style with jitter)
        for i, cat in enumerate(cat_order):
            cat_data = plot_df[plot_df["category"] == cat]["subject_median"].values
            n = len(cat_data)
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=n)
            ax.scatter(
                cat_data,
                i + jitter,
                color="gray",
                alpha=0.5,
                s=12 ** 2,
                zorder=2,
                linewidths=0,
            )

            # Group mean of subject medians with SEM
            group_mean = np.mean(cat_data)
            sem = np.std(cat_data, ddof=1) / np.sqrt(n) if n > 1 else 0.0
            ax.errorbar(
                group_mean,
                i,
                xerr=sem,
                fmt="none",
                ecolor="black",
                elinewidth=2.5,
                capsize=0,
                zorder=4,
            )
            ax.scatter(
                [group_mean],
                [i],
                s=14 ** 2,
                facecolor=COLORS[cat],
                edgecolor="black",
                linewidth=2.5,
                zorder=5,
            )

        # Reference line at 0
        ax.axvline(0, color="black", linestyle="--", linewidth=1.5, zorder=0)

        ax.set_yticks(range(len(cat_order)))
        ax.set_yticklabels(cat_order)
        ax.set_xlabel(f"Subject Median Fixation Advantage ({time_unit_label})")
        ax.set_ylabel("")
        ax.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()

        # Save
        out_pdf = output_dir / _tagged_filename(
            "fixation_advantage_subject_medians", tag=output_tag, ext="pdf"
        )
        out_png = output_dir / _tagged_filename(
            "fixation_advantage_subject_medians", tag=output_tag, ext="png"
        )
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.1)
        fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.1)
        print(f"Saved figure to {out_pdf}")
        print(f"Saved figure to {out_png}")

        if show:
            plt.show()
        else:
            plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    default_base_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Fixation advantage half-violin plot with summary statistics."
    )
    parser.add_argument(
        "--base-dir", type=str, default=str(default_base_dir),
        help="Dataset root containing output/<subid>/.",
    )
    parser.add_argument(
        "--figures-dir", type=str, default="",
        help="Where to write figures/CSVs (default: <base-dir>/figures).",
    )
    parser.add_argument(
        "--exclude-subjects", nargs="*", default=["107", "131"],
        help="Subject IDs to exclude.",
    )
    parser.add_argument(
        "--time-unit-label", type=str, default="ms",
        help="Label for time units (e.g., 'ms' for human, 'steps' for NN sims).",
    )
    parser.add_argument(
        "--output-tag", type=str, default="",
        help="Optional suffix added to output filenames.",
    )
    parser.add_argument(
        "--other-items-mode", type=str, default="all", choices=["all", "seen"],
        help="How to compute mean cumulative time for 'other items'.",
    )
    parser.add_argument(
        "--advantage-timepoint", type=str, default="pre", choices=["pre", "post"],
        help="Whether fixation advantage uses cumulative time up to t-1 ('pre') or including current ('post').",
    )
    parser.add_argument(
        "--exclude-outliers", action="store_true",
        help="Exclude fixation-advantage outliers (mean +/- k*SD).",
    )
    parser.add_argument(
        "--outlier-sd", type=float, default=3.0,
        help="SD multiplier k for outlier exclusion.",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not display interactive plot windows.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    figures_dir = (
        Path(args.figures_dir).resolve()
        if str(args.figures_dir).strip()
        else (base_dir / "figures")
    )
    figures_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        base_dir=base_dir,
        figures_dir=figures_dir,
        excluded_subjects=tuple(str(s) for s in (args.exclude_subjects or [])),
        time_unit_label=str(args.time_unit_label),
        output_tag=str(args.output_tag),
        show_plots=(not bool(args.no_show)),
        other_items_mode=str(args.other_items_mode),
        advantage_timepoint=str(args.advantage_timepoint),
        exclude_outliers=bool(args.exclude_outliers),
        outlier_sd=float(args.outlier_sd),
    )

    # Try data/ first (human layout), fall back to output/ (NN sim layout)
    output_dir = config.base_dir / "data"
    if not any((config.base_dir / "data" / s / f"{s}_fixations_for_modeling.csv").exists()
               for s in _list_numeric_subdirs(config.base_dir / "data")):
        output_dir = config.base_dir / "output"

    # Load fixations
    print("Loading fixations...")
    fixations = load_all_fixations(output_dir, excluded_subjects=config.excluded_subjects)

    # Compute fixation advantage
    print("\nComputing fixation advantage (all/relevant/irrelevant) ...")
    with_adv = compute_fixation_advantages(
        fixations,
        other_items_mode=config.other_items_mode,
        advantage_timepoint=config.advantage_timepoint,
    )

    if config.exclude_outliers:
        k = float(config.outlier_sd)
        print(f"\nExcluding outliers: mean +/- {k:g}*SD ...")
        with_adv["fixation_advantage_all"] = _exclude_outliers_sd(
            with_adv["fixation_advantage_all"], k=k
        )
        with_adv["fixation_advantage_relevant"] = _exclude_outliers_sd(
            with_adv["fixation_advantage_relevant"], k=k
        )
        with_adv["fixation_advantage_irrelevant"] = _exclude_outliers_sd(
            with_adv["fixation_advantage_irrelevant"], k=k
        )

    df_all = _subset_advantage(with_adv, "all")
    df_relevant = _subset_advantage(with_adv, "relevant")
    df_irrelevant = _subset_advantage(with_adv, "irrelevant")

    # Summary statistics table
    print("\nComputing summary statistics...")
    stats_df = compute_summary_stats(df_all, df_relevant, df_irrelevant)
    stats_csv = config.figures_dir / _tagged_filename(
        "fixation_advantage_summary_stats", tag=config.output_tag, ext="csv"
    )
    stats_df.to_csv(stats_csv, index=False)
    print(f"Saved summary statistics to {stats_csv}")
    print(stats_df.to_string(index=False))

    # Plots
    print("\nCreating half-violin plot...")
    plot_half_violin(
        df_all,
        df_relevant,
        df_irrelevant,
        config.figures_dir,
        time_unit_label=config.time_unit_label,
        output_tag=config.output_tag,
        show=config.show_plots,
    )

    print("\nCreating subject medians plot...")
    plot_subject_medians(
        df_all,
        df_relevant,
        df_irrelevant,
        config.figures_dir,
        time_unit_label=config.time_unit_label,
        output_tag=config.output_tag,
        show=config.show_plots,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
