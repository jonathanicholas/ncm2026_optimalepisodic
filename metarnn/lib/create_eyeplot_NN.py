"""Plot NN choice-phase eyegaze summary (proportion relevant fixation time).

Panels:
  1) Proportion of fixation time on offer-relevant items (subject means).
  2) Proportion by decision and reward valence.

Example:
  python metarnn/lib/create_eyeplot_NN.py \\
    --root metarnn/simulations/human_like \\
    --out-dir output/nn_eyegaze
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_subjects(output_root: str) -> List[str]:
    subjects: List[str] = []
    for path in glob.glob(os.path.join(output_root, "*")):
        if os.path.isdir(path):
            base = os.path.basename(path)
            if base.isdigit() and len(base) == 3:
                subjects.append(base)
    subjects.sort()
    return subjects


def _load_fixations(subject: str, root: str, metric: str) -> pd.DataFrame:
    out_dir = os.path.join(root, "data", subject)

    # Try multiple naming conventions for fixation CSVs.
    candidates: List[str] = [
        os.path.join(out_dir, f"{subject}_fixations_df_original.csv"),
        os.path.join(out_dir, f"{subject}_fixations_df_original_buffer_50.csv"),
    ]
    candidates += sorted(glob.glob(os.path.join(out_dir, f"{subject}_fixations_df_original_buffer_*.csv")))

    fix_path = next((p for p in candidates if os.path.exists(p)), "")
    if not fix_path:
        found = sorted(glob.glob(os.path.join(out_dir, "*.csv")))
        raise FileNotFoundError(
            "Missing fixation file. Tried:\n"
            + "\n".join(candidates)
            + ("\n\nFound CSVs:\n" + "\n".join(found) if found else "")
        )

    # Read only what we need (files can be large).
    base_cols = [
        "phase",
        "event",
        "game",
        "trial_number",
        "choice",
        "option",
        "roi_content",
        "true_value",
    ]
    duration_cols = ["fix_duration_bounded", "fix_duration_full"]
    usecols = base_cols + ([] if metric == "count" else duration_cols)

    df = pd.read_csv(fix_path, usecols=lambda c: c in set(usecols))

    # Normalize core types
    for col in ["phase", "event"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()

    for col in ["game", "trial_number", "choice"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "option" in df.columns:
        df["option"] = df["option"].astype(str).str.strip()
    if "roi_content" in df.columns:
        df["roi_content"] = df["roi_content"].astype(str).str.strip()

    if "true_value" in df.columns:
        df["true_value"] = pd.to_numeric(df["true_value"], errors="coerce")

    return df


def _duration_col(df: pd.DataFrame) -> str:
    if "fix_duration_bounded" in df.columns:
        return "fix_duration_bounded"
    if "fix_duration_full" in df.columns:
        return "fix_duration_full"
    raise ValueError("No fixation duration column found.")


def _is_image_row(roi_content: pd.Series) -> pd.Series:
    s = roi_content.astype(str)
    return (~s.isin(["fixation", "none"])) & s.str.contains("_", regex=False)


def _build_valence_map(log_df: pd.DataFrame) -> Dict[tuple, str]:
    """Return {(game, image): valence_str} from the encoding phase of the logfile."""
    enc = log_df[(log_df["phase"] == "encoding") & (log_df["event"] == "image")]
    mapping: Dict[tuple, str] = {}
    for _, row in enc.iterrows():
        game = int(row["game"])
        image = str(row["image"])
        try:
            val = float(row["outcome"])
        except (TypeError, ValueError):
            val = np.nan
        if np.isnan(val):
            valence = "neutral"
        elif val > 0:
            valence = "positive"
        elif val < 0:
            valence = "negative"
        else:
            valence = "neutral"
        mapping[(game, image)] = valence
    return mapping


def _compute_trial_level_proportions(
    df: pd.DataFrame, metric: str, valence_map: Optional[Dict[tuple, str]] = None
) -> pd.DataFrame:
    """Compute trial-level relevant/valence proportions from fixation rows.

    Returns one row per (subject, game, trial_number) with:
      decision_label, prop_relevant, rel_pos, rel_neg

    All proportions are relative to *total image fixation time* per trial.
    """

    d = df[(df["phase"] == "choice") & (df["event"] == "choice")].copy()
    if len(d) == 0:
        return pd.DataFrame([])

    d = d.dropna(subset=["game", "trial_number", "choice"]).copy()
    d = d[d["choice"].isin([1, 2])].copy()

    d["decision_label"] = np.where(d["choice"] == 1, "take", "leave")

    # Keep only image fixations
    d = d[_is_image_row(d["roi_content"])].copy()
    if len(d) == 0:
        return pd.DataFrame([])

    # Relevance: offered token appears in the image name.
    def _relevance_rowwise(roi: str, opt: str) -> bool:
        if not isinstance(opt, str) or opt.strip() == "" or opt.lower() == "nan":
            return False
        parts = str(roi).split("_")
        return opt in parts or opt in str(roi)

    d["relevant"] = [
        _relevance_rowwise(r, o) for r, o in zip(d["roi_content"].astype(str), d["option"].astype(str))
    ]

    # Valence: use per-item reward from logfile if available, else fall back to
    # true_value (which may be the offer value, not the item reward).
    if valence_map is not None:
        d["valence_label"] = [
            valence_map.get((int(g), str(roi)), "neutral")
            for g, roi in zip(d["game"], d["roi_content"])
        ]
    else:
        d["valence_label"] = np.where(
            d["true_value"] > 0, "positive",
            np.where(d["true_value"] < 0, "negative", "neutral")
        )

    if metric == "count":
        d["fix_time"] = 1.0
    else:
        dur_col = _duration_col(d)
        d["fix_time"] = pd.to_numeric(d[dur_col], errors="coerce")
        d = d.dropna(subset=["fix_time"]).copy()

    group_cols = ["subject", "game", "trial_number", "decision_label"]

    # Total image fixation time per trial
    total = d.groupby(group_cols, sort=True)["fix_time"].sum().rename("total").reset_index()

    # Relevant time
    rel = (
        d[d["relevant"]]
        .groupby(group_cols, sort=True)["fix_time"]
        .sum()
        .rename("rel_time")
        .reset_index()
    )

    # Relevant positive/negative time (ignore neutrals)
    rel_pos = (
        d[(d["relevant"]) & (d["valence_label"] == "positive")]
        .groupby(group_cols, sort=True)["fix_time"]
        .sum()
        .rename("rel_pos_time")
        .reset_index()
    )
    rel_neg = (
        d[(d["relevant"]) & (d["valence_label"] == "negative")]
        .groupby(group_cols, sort=True)["fix_time"]
        .sum()
        .rename("rel_neg_time")
        .reset_index()
    )

    out = total.merge(rel, on=group_cols, how="left").merge(rel_pos, on=group_cols, how="left").merge(rel_neg, on=group_cols, how="left")
    out[["rel_time", "rel_pos_time", "rel_neg_time"]] = out[["rel_time", "rel_pos_time", "rel_neg_time"]].fillna(0.0)

    out["prop_relevant"] = np.where(out["total"] > 0, out["rel_time"] / out["total"], np.nan)
    out["rel_pos"] = np.where(out["total"] > 0, out["rel_pos_time"] / out["total"], np.nan)
    out["rel_neg"] = np.where(out["total"] > 0, out["rel_neg_time"] / out["total"], np.nan)

    return out


def compute_eyeplot_nn_tables(
    root: str,
    metric: str = "duration",
    *,
    cache_trial_csv: Optional[str] = None,
    use_cache: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute (or load) the tables needed for the two Figure3_NN panels.

    Returns:
      trial_df: row per (subject, game, trial_number)
      subj_rel: subject means of prop_relevant, with column 'mean_prop'
      subj_val_long: subject means of rel_pos/rel_neg by decision, in long form

    Caching:
      If cache_trial_csv exists and use_cache=True, it is read as trial_df.
      If computed and cache_trial_csv is provided, trial_df is saved there.
    """

    root = os.path.abspath(root)

    trial_df: Optional[pd.DataFrame] = None
    if cache_trial_csv is not None and use_cache and os.path.exists(cache_trial_csv):
        trial_df = pd.read_csv(cache_trial_csv)

    if trial_df is None:
        subjects = find_subjects(os.path.join(root, "data"))
        if len(subjects) == 0:
            raise RuntimeError(f"No subjects found under {os.path.join(root, 'data')}")

        all_trials = []
        for sid in subjects:
            df_fix = _load_fixations(sid, root, metric)
            df_fix["subject"] = str(sid)
            # Build per-item valence map from logfile encoding phase
            log_path = os.path.join(root, "data", sid, f"{sid}_MAIN_logfile_7.csv")
            valence_map = None
            if os.path.exists(log_path):
                log_df = pd.read_csv(log_path)
                for col in ["phase", "event"]:
                    if col in log_df.columns:
                        log_df[col] = log_df[col].astype(str).str.lower().str.strip()
                valence_map = _build_valence_map(log_df)
            trials = _compute_trial_level_proportions(df_fix, metric, valence_map=valence_map)
            if len(trials) == 0:
                continue
            all_trials.append(trials)

        if len(all_trials) == 0:
            raise RuntimeError("No choice-phase fixation trials found.")

        trial_df = pd.concat(all_trials, ignore_index=True)

        if cache_trial_csv is not None:
            ensure_output_dir(os.path.dirname(cache_trial_csv) or ".")
            trial_df.to_csv(cache_trial_csv, index=False)

    # Panel 1: relevant-only subject means
    subj_rel = (
        trial_df.groupby("subject")["prop_relevant"]
        .mean()
        .reset_index()
        .rename(columns={"prop_relevant": "mean_prop"})
    )

    # Panel 2: subject means for relevant positive/negative by decision
    subj_val = (
        trial_df.groupby(["subject", "decision_label"])[["rel_pos", "rel_neg"]]
        .mean()
        .reset_index()
    )
    subj_val_long = subj_val.melt(
        id_vars=["subject", "decision_label"],
        value_vars=["rel_pos", "rel_neg"],
        var_name="category",
        value_name="mean_prop",
    )
    subj_val_long["valence_label"] = np.where(subj_val_long["category"] == "rel_pos", "positive", "negative")

    return trial_df, subj_rel, subj_val_long


def _panel_relevant_only(
    ax: plt.Axes,
    subj_means: pd.DataFrame,
    *,
    strip_color: str = "gray",
    mean_facecolor: str = ".5",
    chance_style: str = "k--",
    chance_linewidth: float | None = None,
    strip_zorder: int = 0,
    strip_edgecolor: str | None = "black",
    strip_linewidth: float = 1,
) -> None:
    """Match styling of `add_panel_B_relevant_only` from analyze_eyetracking.py."""

    vals = subj_means["mean_prop"].astype(float).to_numpy()

    # Place seed points at jittered offsets away from x=0 to avoid overlap with group mean.
    # Alternate signs deterministically so points split across both sides of the group mean
    # (e.g., for n=5 this gives 3 on the left and 2 on the right).
    n = len(vals)
    rng = np.random.default_rng(42)
    magnitudes = rng.uniform(0.015, 0.045, size=n)
    signs = np.where(np.arange(n) % 2 == 0, -1.0, 1.0)
    offsets = magnitudes * signs
    scatter_kw: dict = dict(
        s=12**2, facecolor=strip_color, alpha=0.5, linewidth=strip_linewidth, zorder=strip_zorder,
    )
    if strip_edgecolor is not None:
        scatter_kw["edgecolor"] = strip_edgecolor
    else:
        scatter_kw["edgecolor"] = "none"
    ax.scatter(offsets, vals, **scatter_kw)

    mean_val = float(np.nanmean(vals)) if len(vals) else np.nan
    if len(vals) > 1:
        se = float(np.nanstd(vals, ddof=1)) / np.sqrt(np.sum(~np.isnan(vals)))
        ci = 1.96 * se
    else:
        ci = 0.0

    ax.errorbar([0], [mean_val], yerr=[ci], fmt="none", ecolor="black", capsize=0, zorder=3)
    ax.scatter(
        [0],
        [mean_val],
        s=14 ** 2,
        facecolor=mean_facecolor,
        edgecolor="black",
        linewidth=2.5,
        zorder=3,
    )

    ax.set_ylim(0.2, 0.8)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels([0.2, 0.4, 0.6, 0.8])
    ax.set_ylabel("Prop. Relevant Fix. Time")

    ax.set_xticks([])
    ax.set_xlim(-0.075, 0.075)
    chance_kw: dict = {}
    if chance_linewidth is not None:
        chance_kw["linewidth"] = chance_linewidth
    ax.plot((-0.075, 0.075), (0.5, 0.5), chance_style, **chance_kw)

    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def _panel_take_leave_valence(
    ax: plt.Axes,
    df_subj_long: pd.DataFrame,
    *,
    lighten_amount: float = 0.0,
    subject_line_color: str = "0.7",
    figure3_legend: bool = False,
) -> None:
    """Match styling of `add_panel_C_relsign4_relevant` from analyze_eyetracking.py.

    Parameters
    ----------
    lighten_amount : float
        If > 0, blend dot face-colors toward white by this fraction (0 = original, 1 = white).
    subject_line_color : str
        Color for thin per-subject lines.
    figure3_legend : bool
        If True, use the Figure3 legend style (frameon, edgecolor, labels with "Reward").
    """
    import matplotlib.colors as mcolors

    def _lighten(color, amount):
        c = np.array(mcolors.to_rgb(color))
        return tuple(1 - amount * (1 - c))

    required_cols = {"subject", "decision_label", "valence_label", "mean_prop"}
    missing = required_cols - set(df_subj_long.columns)
    if missing:
        raise ValueError(f"Missing columns for take/leave panel: {missing}")

    decision_order = [d for d in ["take", "leave"] if d in df_subj_long["decision_label"].unique()] or sorted(df_subj_long["decision_label"].unique())
    valence_order = [v for v in ["positive", "negative"] if v in df_subj_long["valence_label"].unique()] or sorted(df_subj_long["valence_label"].unique())

    palette = sns.color_palette("deep", n_colors=len(valence_order))
    x_index = {d: i for i, d in enumerate(decision_order)}

    offsets_map = {1: [0.0], 2: [-0.32, 0.32], 3: [-0.2, 0.0, 0.2]}
    offsets = offsets_map.get(len(valence_order), np.linspace(-0.3, 0.3, len(valence_order)))
    val_to_offset = {vlab: offsets[j] for j, vlab in enumerate(valence_order)}

    df_subj_long = df_subj_long.copy()
    df_subj_long["subject"] = df_subj_long["subject"].astype(str)

    # Thin subject lines
    for subj in df_subj_long["subject"].unique():
        for dlab in decision_order:
            xs, ys = [], []
            for vlab in valence_order:
                row = df_subj_long[
                    (df_subj_long["subject"] == subj)
                    & (df_subj_long["decision_label"] == dlab)
                    & (df_subj_long["valence_label"] == vlab)
                ]
                if not row.empty:
                    xs.append(x_index[dlab] + val_to_offset[vlab])
                    ys.append(float(row["mean_prop"].values[0]))
            if len(xs) >= 2:
                ax.plot(xs, ys, color=subject_line_color, alpha=0.8, linewidth=0.8, zorder=1)

    stats_df = (
        df_subj_long.groupby(["decision_label", "valence_label"])["mean_prop"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats_df["sem"] = stats_df["std"] / np.sqrt(stats_df["count"].clip(lower=1))
    if lighten_amount > 0:
        color_map = {vlab: _lighten(palette[j], lighten_amount) for j, vlab in enumerate(valence_order)}
    else:
        color_map = {vlab: palette[j] for j, vlab in enumerate(valence_order)}

    for dlab in decision_order:
        rows = stats_df[stats_df["decision_label"] == dlab]
        rows = rows.set_index("valence_label").reindex(valence_order).reset_index()
        xs = [x_index[dlab] + val_to_offset[v] for v in rows["valence_label"]]
        ys = rows["mean"].values.astype(float)
        ses = rows["sem"].values.astype(float)

        ax.plot(xs, ys, color="black", linewidth=2.5, zorder=5)
        for xi, yi, sei, vlab in zip(xs, ys, ses, rows["valence_label"]):
            ax.errorbar(
                xi,
                yi,
                yerr=sei * 2,
                fmt="o",
                ms=14,
                mfc=color_map[vlab],
                mec="black",
                mew=2.5,
                ecolor="black",
                elinewidth=2.5,
                capsize=0,
                zorder=6,
            )

    ax.set_ylabel("Proportion Fixation Time")
    ax.set_xlabel("")

    from matplotlib.lines import Line2D

    if figure3_legend:
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_map[v],
                markeredgecolor="black",
                markeredgewidth=2.5,
                markersize=14,
                linewidth=0,
            )
            for v in valence_order
        ]
        label_map = {"positive": "Positive Reward", "negative": "Negative Reward"}
        ax.legend(
            legend_handles,
            [label_map.get(v, v) for v in valence_order],
            fontsize=20,
            handletextpad=0.3,
            loc="upper center",
            frameon=True,
            edgecolor="black",
            fancybox=False,
        )
    else:
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                markerfacecolor=color_map[v],
                markeredgecolor="black",
                markersize=8,
                linewidth=0,
            )
            for v in valence_order
        ]
        label_map = {"positive": "Positive", "negative": "Negative"}
        ax.legend(
            legend_handles,
            [label_map.get(v, v) for v in valence_order],
            title="Reward Valence",
            fontsize=20,
            title_fontsize=20,
            frameon=False,
        )

    dec_label_map = {"take": "Take", "leave": "Leave"}
    ax.set_xticks([x_index[d] for d in decision_order])
    ax.set_xticklabels([dec_label_map.get(d, d) for d in decision_order])

    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_ylim(0, 0.63)

    sns.despine(ax=ax)


def create_figure3_nn(root: str, out_dir: str, metric: str = "duration") -> str:
    ensure_output_dir(out_dir)

    cache_trial_csv = os.path.join(out_dir, f"Figure3_NN_trial_level_{metric}.csv")
    _, subj_rel, subj_val_long = compute_eyeplot_nn_tables(
        root=root,
        metric=metric,
        cache_trial_csv=cache_trial_csv,
        use_cache=True,
    )

    sns.set_context("poster")
    with plt.rc_context({
        "font.family": "Arial",
        "axes.titlesize": 24,
        "axes.labelsize": 28,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
    }):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1, 2]})
        _panel_relevant_only(ax1, subj_rel)
        _panel_take_leave_valence(ax2, subj_val_long)

        # Panel labels
        ax1.text(-0.35, 1.1, "A", transform=ax1.transAxes, fontsize=26, fontweight="bold", ha="left", va="top")
        ax2.text(-0.15, 1.1, "B", transform=ax2.transAxes, fontsize=26, fontweight="bold", ha="left", va="top")

        fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.14, wspace=0.35)

        out_path = os.path.join(out_dir, "Figure3_NN.pdf")
        # Use a tight bounding box so y-axis labels/ticks (Panel A) are not clipped.
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create minimal NN eyegaze Figure3 (choice-only panels).")
    parser.add_argument("--root", default="metarnn/simulations/human_like", help="Root containing NN human_like data/output folders.")
    parser.add_argument("--out-dir", default="output/nn_eyegaze", help="Output directory for the PDF.")
    parser.add_argument("--metric", choices=["duration", "count"], default="duration", help="Use fixation duration or count.")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    out_dir = os.path.abspath(args.out_dir)

    out_path = create_figure3_nn(root=root, out_dir=out_dir, metric=args.metric)
    print(f"Saved NN Figure3 eyegaze PDF to {out_path}")


if __name__ == "__main__":
    main()
