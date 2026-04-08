#!/usr/bin/env python3
"""Create a Humans vs NN comparison figure (panels A-H, 2x4 layout).

Panels: A) first-fix vs revisit relevance, B) cumulative fixation-time curve,
C-D) fixation duration by position (human / NN), E) fixation count by relevance,
F-H) proportion relevant and valence diff by fixation position.

Requires upstream revisit CSVs from compute_revisits_count_and_duration.py.

Example:
  conda run -n analysis python metarnn/lib/plot_NN_H_comparison.py \\
    --nn-root metarnn/simulations/human_like_04_04_input5 \\
    --out-dir metarnn/simulations/human_like_04_04_input5/output/human_comparison \\
    --tag 04_04_input5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from analysis.lib.analyze_fixation_duration_by_position import (  # noqa: E402
    COLLAPSE_KEEP_FIRST_N,
    DEFAULT_EXCLUDED as DEFAULT_EXCLUDED_EYE,
    build_duration_summaries_7plus,
    load_true_rewards,
    load_clean_choice_fixations as load_clean_choice_fixations_full,
)
from analysis.lib.visualize_first_fixations_relevance_and_magnitude import (  # noqa: E402
    load_recalled_rewards,
)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return float("nan")
    return float(np.nanstd(x, ddof=1) / np.sqrt(len(x)))


def _add_panel_label(ax: plt.Axes, label: str, *, dx: float = -55, dy: float = 14) -> None:
    ax.annotate(
        label,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=26,
        fontweight="bold",
        ha="left",
        va="top",
        annotation_clip=False,
    )


def _add_shared_xlabel(fig: plt.Figure, left_ax: plt.Axes, right_ax: plt.Axes, label: str, *, dy: float = 0.035) -> None:
    """Add a single x-axis label centered under two side-by-side axes."""

    b0 = left_ax.get_position()
    b1 = right_ax.get_position()
    x_center = (b0.x0 + b1.x1) / 2.0
    y = min(b0.y0, b1.y0) - float(dy)
    fig.text(x_center, y, label, ha="center", va="top", fontsize=22)


def _find_revisits_by_subject_csv(stats_dir: Path, *, agent_tag: str, tag: str = "") -> Path:
    """Locate the revisits by-subject CSV under *stats_dir*.

    Human files live at ``REPO_ROOT/output/eyegaze/stats/`` and are named
    without an output-tag suffix (e.g. ``..._human.csv``).
    NN files live at ``OUT_ROOT/output/eyegaze/stats/`` and may include a
    tag suffix (e.g. ``..._nn_<tag>.csv``).
    """
    # Try with tag first, then without.
    if tag:
        path = stats_dir / f"revisits_count_and_duration_by_subject_{agent_tag}_{tag}.csv"
        if path.exists():
            return path
    path = stats_dir / f"revisits_count_and_duration_by_subject_{agent_tag}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing revisits by-subject CSV: looked for "
            f"revisits_count_and_duration_by_subject_{agent_tag}[_{tag}].csv "
            f"under {stats_dir}"
        )
    return path


def _find_default_clean_choice_fixations(base_dir: Path) -> Path:
    out_dir = base_dir / "output"
    preferred = out_dir / "choice_fixations_clean_buffer_50.csv"
    if preferred.exists():
        return preferred
    fallback = out_dir / "choice_fixations_clean.csv"
    if fallback.exists():
        return fallback
    matches = sorted(out_dir.glob("choice_fixations_clean_buffer_*.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find choice_fixations_clean*.csv under {out_dir}")


def _load_clean_choice_fixations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [
        "subject_id",
        "game",
        "trial_number",
        "option",
        "fixation_count",
        "fixation_duration",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}\nfile={path}")

    df = df.copy()
    df["subject_id"] = df["subject_id"].astype(str)
    df["game"] = pd.to_numeric(df["game"], errors="coerce")
    df["trial_number"] = pd.to_numeric(df["trial_number"], errors="coerce")
    df["fixation_count"] = pd.to_numeric(df["fixation_count"], errors="coerce")
    df["fixation_duration"] = pd.to_numeric(df["fixation_duration"], errors="coerce")
    df = df.dropna(subset=["game", "trial_number", "fixation_count", "fixation_duration"]).copy()
    df["game"] = df["game"].astype(int)
    df["trial_number"] = df["trial_number"].astype(int)
    df["fixation_count"] = df["fixation_count"].astype(int)

    df = df[df["fixation_duration"] > 0].copy()
    return df


def _panel_duration_by_relevance_7plus(
    ax: plt.Axes,
    *,
    base_dir: Path,
    clean_choice_fixations_path: Optional[Path] = None,
    title: str,
    y_label: str = "Fixation duration (ms)",
    y_scale: float = 1.0,
    y_lim: Optional[Tuple[float, float]] = None,
    max_fixations: int,
    keep_first_n: int,
    excluded_subjects: Tuple[str, ...] = tuple(DEFAULT_EXCLUDED_EYE),
    show_grid: bool = False,
    colors: Optional[dict] = None,
    fix_df: Optional[pd.DataFrame] = None,
    precomputed_summary: Optional[pd.DataFrame] = None,
) -> None:
    if precomputed_summary is not None:
        summary_7p = precomputed_summary
    else:
        if fix_df is not None:
            fix = fix_df
        else:
            fix = load_clean_choice_fixations_full(clean_choice_fixations_path)
            excluded = set(map(str, excluded_subjects))
            if excluded:
                fix = fix[~fix["subject_id"].isin(excluded)].copy()
        if fix.empty:
            raise RuntimeError("No fixations left after exclusions")

        summary_7p = build_duration_summaries_7plus(
            fix,
            max_fixations=int(max_fixations),
            base_dir=base_dir,
            excluded_subjects=tuple(excluded_subjects),
            keep_first_n=int(keep_first_n),
        )
    if summary_7p is None or summary_7p.empty:
        raise RuntimeError("Duration summary is empty")

    # Mirror plot_duration_summaries() panel 2 (relevant vs irrelevant).
    legend_map = {"relevant": "Relevant", "irrelevant": "Irrelevant"}
    _colors = colors or {"relevant": "tab:blue", "irrelevant": "tab:orange"}
    for subset_name, color in [("relevant", _colors["relevant"]), ("irrelevant", _colors["irrelevant"])]:
        s = (
            summary_7p[(summary_7p["panel"] == "by_relevance") & (summary_7p["subset"] == subset_name)]
            .copy()
            .sort_values("fixation_position")
        )
        if s.empty:
            continue
        x = pd.to_numeric(s["fixation_position"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(s["mean_duration"], errors="coerce").to_numpy(dtype=float)
        se = pd.to_numeric(s.get("mean_duration_sem"), errors="coerce").to_numpy(dtype=float)

        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]
        y = y[ok]
        se = se[ok] if len(se) == len(ok) else se

        y_plot = y * float(y_scale)
        se_plot = se * float(y_scale) if (np.isfinite(se).any() and len(se) == len(y)) else se

        ax.plot(x, y_plot, color=color, linewidth=3.0, label=legend_map.get(subset_name, subset_name))
        if np.isfinite(se).any() and len(se) == len(y):
            ax.fill_between(x, y_plot - se_plot, y_plot + se_plot, color=color, alpha=0.15, linewidth=0)

    ax.set_title(title)
    ax.set_xlabel("Fixation number")
    ax.set_ylabel(str(y_label))
    if y_lim is not None:
        ax.set_ylim(float(y_lim[0]), float(y_lim[1]))

    # 7+ collapsed x-axis labeling: 1..keep_first_n plus tail bin.
    ticks = list(range(1, int(keep_first_n) + 2))
    ax.set_xticks(ticks)
    tail_label = f"{int(keep_first_n) + 1}+"
    ax.set_xticklabels([str(i) for i in range(1, int(keep_first_n) + 1)] + [tail_label])
    if show_grid:
        ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=10, loc="best")
    ax.spines[["top", "right"]].set_visible(False)


def _compute_prop_relevant_by_position_7plus(
    fix: pd.DataFrame,
    *,
    max_fixations: int,
    keep_first_n: int,
) -> pd.DataFrame:
    """Compute mean±SEM across subjects of P(relevant | fixation position), with a 7+ tail bin.

    Uses a within-trial aggregation for the tail bin so trials with many tail fixations
    do not dominate the 7+ point.
    """

    if fix is None or fix.empty:
        return pd.DataFrame()

    required = {"subject_id", "game", "trial_number", "option", "fixation_count", "is_relevant"}
    missing = sorted(required - set(fix.columns))
    if missing:
        raise ValueError(f"Fixations missing required columns for prop-relevant plot: {missing}")

    df = fix.copy()
    df["fixation_count"] = pd.to_numeric(df["fixation_count"], errors="coerce")
    df = df.dropna(subset=["fixation_count"]).copy()
    df["fixation_count"] = df["fixation_count"].astype(int)
    df = df[df["fixation_count"] <= int(max_fixations)].copy()
    if df.empty:
        return pd.DataFrame()

    trial_cols = ["subject_id", "game", "trial_number", "option"]
    df["fixation_position"] = df["fixation_count"].astype(int)
    df["is_relevant"] = pd.to_numeric(df["is_relevant"], errors="coerce").fillna(0).astype(int)

    early = df[df["fixation_position"] <= int(keep_first_n)].copy()
    early["prop_relevant"] = early["is_relevant"].astype(float)
    early = early[trial_cols + ["fixation_position", "prop_relevant"]]

    tail = df[df["fixation_position"] > int(keep_first_n)].copy()
    if not tail.empty:
        tail_agg = tail.groupby(trial_cols, as_index=False).agg(prop_relevant=("is_relevant", "mean"))
        tail_agg["fixation_position"] = int(keep_first_n) + 1
        tail_agg = tail_agg[trial_cols + ["fixation_position", "prop_relevant"]]
        collapsed = pd.concat([early, tail_agg], ignore_index=True)
    else:
        collapsed = early

    if collapsed.empty:
        return pd.DataFrame()

    # Per-subject means first.
    per_sub = (
        collapsed.groupby(["subject_id", "fixation_position"], as_index=False)
        .agg(prop_relevant=("prop_relevant", "mean"))
        .copy()
    )

    def _sem_series(x: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce").dropna()
        if len(x) <= 1:
            return float("nan")
        return float(x.std(ddof=1) / np.sqrt(len(x)))

    summary = per_sub.groupby("fixation_position", as_index=False).agg(
        prop_relevant_mean=("prop_relevant", "mean"),
        prop_relevant_sem=("prop_relevant", _sem_series),
        n_subjects=("subject_id", lambda s: int(pd.Series(s).nunique())),
    )
    summary["fixation_position"] = pd.to_numeric(summary["fixation_position"], errors="coerce").astype(int)
    summary = summary.sort_values("fixation_position").reset_index(drop=True)
    return summary


def _panel_prop_relevant_fixated_by_position_7plus(
    ax: plt.Axes,
    *,
    base_dir: Path,
    clean_choice_fixations_path: Optional[Path] = None,
    title: str,
    max_fixations: int,
    keep_first_n: int,
    excluded_subjects: Tuple[str, ...] = tuple(DEFAULT_EXCLUDED_EYE),
    y_label: str = "Prop. relevant fixated",
    hline_y: float = 0.5,
    line_color: str = "#1f77b4",
    band_alpha: float = 0.20,
    y_lim: Tuple[float, float] = (0.0, 1.0),
    fix_df: Optional[pd.DataFrame] = None,
    precomputed_summary: Optional[pd.DataFrame] = None,
) -> None:
    if precomputed_summary is not None:
        summary = precomputed_summary
    else:
        if fix_df is not None:
            fix = fix_df
        else:
            fix = load_clean_choice_fixations_full(clean_choice_fixations_path)
            excluded = set(map(str, excluded_subjects))
            if excluded:
                fix = fix[~fix["subject_id"].isin(excluded)].copy()
        if fix.empty:
            raise RuntimeError("No fixations left after exclusions")

        summary = _compute_prop_relevant_by_position_7plus(
            fix,
            max_fixations=int(max_fixations),
            keep_first_n=int(keep_first_n),
        )
    if summary is None or summary.empty:
        raise RuntimeError("Prop-relevant summary is empty")

    x = pd.to_numeric(summary["fixation_position"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(summary["prop_relevant_mean"], errors="coerce").to_numpy(dtype=float)
    se = pd.to_numeric(summary.get("prop_relevant_sem"), errors="coerce").to_numpy(dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    se = se[ok] if len(se) == len(ok) else se

    ax.plot(x, y, color=str(line_color), linewidth=2.5)
    if np.isfinite(se).any() and len(se) == len(y):
        lo = np.clip(y - se, 0.0, 1.0)
        hi = np.clip(y + se, 0.0, 1.0)
        ax.fill_between(x, lo, hi, color=str(line_color), alpha=float(band_alpha), linewidth=0)
    ax.axhline(float(hline_y), color="black", linewidth=1.5, linestyle=":")

    ax.set_title(title)
    ax.set_xlabel("Fixation number")
    ax.set_ylabel(str(y_label))
    ax.set_ylim(float(y_lim[0]), float(y_lim[1]))

    ticks = list(range(1, int(keep_first_n) + 2))
    ax.set_xticks(ticks)
    tail_label = f"{int(keep_first_n) + 1}+"
    ax.set_xticklabels([str(i) for i in range(1, int(keep_first_n) + 1)] + [tail_label])

    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)


def _rank_abs_value_bins_1_highest(values: pd.Series) -> pd.Series:
    """Return rank bins 1..3 where 1 is the highest |value|.

    Tie rule: ties are assigned to the *best* rank they qualify for.
    Concretely, we rank |value| in descending order using rank(method='min').
    """

    v = pd.to_numeric(values, errors="coerce").abs()
    r = v.rank(method="min", ascending=False)
    r = pd.to_numeric(r, errors="coerce").fillna(np.nan)
    return r.clip(lower=1, upper=3)


def _build_relevant_item_rank_map_and_chance(
    *,
    base_dir: Path,
    fix: pd.DataFrame,
    excluded_subjects: Tuple[str, ...],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return:
    1) (subject_id, game, option, image) -> rank_bin (1..3; 1 highest |value|)
    2) (subject_id, game, option) -> chance_prop_bin{1,2,3}
    """

    if fix.empty:
        return (
            pd.DataFrame(columns=["subject_id", "game", "option", "image", "rank_bin"]),
            pd.DataFrame(columns=["subject_id", "game", "option", "chance_prop_bin1", "chance_prop_bin2", "chance_prop_bin3"]),
        )

    subjects = sorted(fix["subject_id"].dropna().astype(str).unique().tolist())
    true_rewards = load_true_rewards(base_dir, subjects=subjects, excluded_subjects=tuple(excluded_subjects))
    if true_rewards is None or true_rewards.empty:
        raise RuntimeError(f"No true rewards found under {base_dir}; cannot compute abs-value bins.")

    # Determine relevant items for each (subject, game, option) using the token rule: option must be a token in image.
    opts = fix[["subject_id", "game", "option"]].drop_duplicates().copy()
    opt_items = opts.merge(true_rewards, on=["subject_id", "game"], how="left")
    opt_items = opt_items.dropna(subset=["image", "reward_true"]).copy()
    opt_items["image"] = opt_items["image"].astype(str)
    opt_items["option"] = opt_items["option"].astype(str)

    opt_items["is_relevant_token"] = [
        int(isinstance(opt, str) and isinstance(img, str) and opt in img.split("_"))
        for opt, img in zip(opt_items["option"].tolist(), opt_items["image"].tolist())
    ]

    rel = opt_items[opt_items["is_relevant_token"] == 1].copy()
    if rel.empty:
        raise RuntimeError("No relevant items found when computing abs-value bins.")

    rel["rank_bin"] = rel.groupby(["subject_id", "game", "option"], sort=False)["reward_true"].transform(
        _rank_abs_value_bins_1_highest
    )
    rel["rank_bin"] = pd.to_numeric(rel["rank_bin"], errors="coerce").astype("Int64")
    rel = rel.dropna(subset=["rank_bin"]).copy()
    rel["rank_bin"] = rel["rank_bin"].astype(int)
    rel["rank_bin"] = rel["rank_bin"].clip(lower=1, upper=3)

    rank_map = rel[["subject_id", "game", "option", "image", "rank_bin"]].drop_duplicates(
        subset=["subject_id", "game", "option", "image"]
    )

    # Chance: within each (subject, game, option), fraction of the 3 relevant items in each bin.
    counts = rank_map.groupby(["subject_id", "game", "option", "rank_bin"], as_index=False).agg(n_items=("image", "size"))
    chance = (
        counts.pivot_table(index=["subject_id", "game", "option"], columns="rank_bin", values="n_items", fill_value=0)
        .reset_index()
        .rename(columns={1: "n_bin1", 2: "n_bin2", 3: "n_bin3"})
    )
    for col in ["n_bin1", "n_bin2", "n_bin3"]:
        if col not in chance.columns:
            chance[col] = 0
    chance["chance_prop_bin1"] = pd.to_numeric(chance["n_bin1"], errors="coerce").fillna(0.0) / 3.0
    chance["chance_prop_bin2"] = pd.to_numeric(chance["n_bin2"], errors="coerce").fillna(0.0) / 3.0
    chance["chance_prop_bin3"] = pd.to_numeric(chance["n_bin3"], errors="coerce").fillna(0.0) / 3.0
    chance = chance[["subject_id", "game", "option", "chance_prop_bin1", "chance_prop_bin2", "chance_prop_bin3"]]

    return rank_map, chance


def _panel_prop_relevant_fixations_within_absvalue_bin_by_position_7plus(
    ax: plt.Axes,
    *,
    base_dir: Path,
    clean_choice_fixations_path: Path,
    title: str,
    max_fixations: int,
    keep_first_n: int,
    excluded_subjects: Tuple[str, ...] = tuple(DEFAULT_EXCLUDED_EYE),
    y_lim: Tuple[float, float] = (0.2, 0.5),
) -> None:
    fix = load_clean_choice_fixations_full(clean_choice_fixations_path)
    excluded = set(map(str, excluded_subjects))
    if excluded:
        fix = fix[~fix["subject_id"].isin(excluded)].copy()
    if fix.empty:
        raise RuntimeError(f"No fixations left after exclusions for: {clean_choice_fixations_path}")

    # Map each relevant fixation to its (1..3) abs-value rank bin among relevant items.
    rank_map, chance = _build_relevant_item_rank_map_and_chance(
        base_dir=base_dir,
        fix=fix,
        excluded_subjects=tuple(excluded_subjects),
    )

    d = fix.copy()
    d["fixation_count"] = pd.to_numeric(d["fixation_count"], errors="coerce")
    d = d.dropna(subset=["fixation_count"]).copy()
    d["fixation_count"] = d["fixation_count"].astype(int)
    d = d[d["fixation_count"] <= int(max_fixations)].copy()
    d = d[d["is_relevant"] == 1].copy()
    if d.empty:
        raise RuntimeError(f"No relevant fixations available for: {clean_choice_fixations_path}")

    d = d.merge(rank_map, on=["subject_id", "game", "option", "image"], how="left")
    d["rank_bin"] = pd.to_numeric(d["rank_bin"], errors="coerce")
    d = d.dropna(subset=["rank_bin"]).copy()
    d["rank_bin"] = d["rank_bin"].astype(int)
    d = d[d["rank_bin"].isin([1, 2, 3])].copy()

    trial_cols = ["subject_id", "game", "trial_number", "option"]
    d["fixation_position"] = d["fixation_count"].astype(int)
    d["pos_collapsed"] = np.where(
        d["fixation_position"] <= int(keep_first_n),
        d["fixation_position"],
        int(keep_first_n) + 1,
    ).astype(int)

    # Per-trial proportions (to avoid overweighting trials with many tail fixations).
    # IMPORTANT: include explicit zeros for bins that receive no fixations
    # on a given (trial, position), so the bin proportions sum to 1.
    counts = d.groupby(trial_cols + ["pos_collapsed", "rank_bin"], as_index=False).agg(n_fix=("rank_bin", "size"))
    totals = d.groupby(trial_cols + ["pos_collapsed"], as_index=False).agg(n_total=("rank_bin", "size"))
    totals = totals[totals["n_total"] > 0].copy()

    bins = pd.DataFrame({"rank_bin": [1, 2, 3]})
    grid = totals.assign(_k=1).merge(bins.assign(_k=1), on="_k", how="left").drop(columns=["_k"])
    counts = grid.merge(counts, on=trial_cols + ["pos_collapsed", "rank_bin"], how="left")
    counts["n_fix"] = pd.to_numeric(counts["n_fix"], errors="coerce").fillna(0.0)
    counts["prop"] = counts["n_fix"] / counts["n_total"]

    # Within-subject means over trials.
    per_sub = counts.groupby(["subject_id", "pos_collapsed", "rank_bin"], as_index=False).agg(prop=("prop", "mean"))

    def _sem_series(x: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce").dropna()
        if len(x) <= 1:
            return float("nan")
        return float(x.std(ddof=1) / np.sqrt(len(x)))

    summary = per_sub.groupby(["pos_collapsed", "rank_bin"], as_index=False).agg(
        prop_mean=("prop", "mean"),
        prop_sem=("prop", _sem_series),
        n_subjects=("subject_id", lambda s: int(pd.Series(s).nunique())),
    )
    summary["pos_collapsed"] = pd.to_numeric(summary["pos_collapsed"], errors="coerce").astype(int)
    summary["rank_bin"] = pd.to_numeric(summary["rank_bin"], errors="coerce").astype(int)

    # Chance baselines: compute per-subject chance means across trials, then group mean per bin.
    trials = fix[trial_cols].drop_duplicates().copy()
    chance_t = trials.merge(chance, on=["subject_id", "game", "option"], how="left")

    chance_long = chance_t.melt(
        id_vars=["subject_id", "game", "trial_number", "option"],
        value_vars=["chance_prop_bin1", "chance_prop_bin2", "chance_prop_bin3"],
        var_name="chance_bin",
        value_name="chance_prop",
    )
    chance_long["rank_bin"] = chance_long["chance_bin"].map(
        {"chance_prop_bin1": 1, "chance_prop_bin2": 2, "chance_prop_bin3": 3}
    )
    chance_long["chance_prop"] = pd.to_numeric(chance_long["chance_prop"], errors="coerce")
    chance_sub = chance_long.groupby(["subject_id", "rank_bin"], as_index=False).agg(chance_prop=("chance_prop", "mean"))
    chance_sum = chance_sub.groupby("rank_bin", as_index=False).agg(
        chance_mean=("chance_prop", "mean"),
        chance_sem=("chance_prop", _sem_series),
    )

    # Plot settings (purple palette: dark → light for high → low |V|).
    colors = {1: "#54278f", 2: "#756bb1", 3: "#9e9ac8"}
    labels = {1: "High |V|", 2: "Med. |V|", 3: "Low |V|"}
    band_alpha = 0.20

    for b in [1, 2, 3]:
        s = summary[summary["rank_bin"] == b].copy().sort_values("pos_collapsed")
        if s.empty:
            continue
        x = pd.to_numeric(s["pos_collapsed"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(s["prop_mean"], errors="coerce").to_numpy(dtype=float)
        se = pd.to_numeric(s["prop_sem"], errors="coerce").to_numpy(dtype=float)

        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]
        y = y[ok]
        se = se[ok] if len(se) == len(ok) else se

        ax.plot(x, y, color=colors[b], linewidth=2.5, label=labels[b])
        if np.isfinite(se).any() and len(se) == len(y):
            lo = np.clip(y - se, 0.0, 1.0)
            hi = np.clip(y + se, 0.0, 1.0)
            ax.fill_between(x, lo, hi, color=colors[b], alpha=band_alpha, linewidth=0)

        # Chance baseline for this bin.
        c = chance_sum[chance_sum["rank_bin"] == b]
        if not c.empty:
            chance_y = float(pd.to_numeric(c["chance_mean"].iloc[0], errors="coerce"))
            if np.isfinite(chance_y):
                ax.axhline(chance_y, color=colors[b], linewidth=1.5, linestyle=":")

    ax.set_title(title)
    ax.set_xlabel("Fixation number")
    ax.set_ylabel("Prop. relevant fixated")
    ax.set_ylim(float(y_lim[0]), float(y_lim[1]))

    ticks = list(range(1, int(keep_first_n) + 2))
    ax.set_xticks(ticks)
    tail_label = f"{int(keep_first_n) + 1}+"
    ax.set_xticklabels([str(i) for i in range(1, int(keep_first_n) + 1)] + [tail_label])
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="best")


# ---------------------------------------------------------------------------
# Panel G/H: prop positive vs negative relevant fixations by take/leave
# ---------------------------------------------------------------------------


def _compute_prop_positive_negative_relevant_by_decision(
    fix: pd.DataFrame,
    *,
    max_fixations: int,
    keep_first_n: int,
    decision: int,
    denom: str = "relevant",
    reward_col: str = "reward",
) -> pd.DataFrame:
    """Compute group mean±SEM of proportion positive/negative relevant fixations.

    Parameters
    ----------
    fix : pd.DataFrame
        Must have columns: subject_id, game, trial_number, option,
        fixation_count, is_relevant, reward, choice.
    decision : int
        1 = take, 2 = leave.
    denom : str
        'relevant' — denominator is relevant-item fixations only (sums to 1).
        'all' — denominator is all fixations including irrelevant (like overview fig).
    reward_col : str
        Column to use for valence (sign). Default 'reward' (true reward).
        Use 'reward_recalled' for participant-reported values.

    Returns
    -------
    pd.DataFrame with columns: pos_collapsed, valence, prop_mean, prop_sem.
    """
    d = fix.copy()
    d["choice"] = pd.to_numeric(d["choice"], errors="coerce")
    d = d[d["choice"] == decision].copy()

    d["fixation_count"] = pd.to_numeric(d["fixation_count"], errors="coerce")
    d = d.dropna(subset=["fixation_count"]).copy()
    d["fixation_count"] = d["fixation_count"].astype(int)
    d = d[d["fixation_count"] <= int(max_fixations)].copy()

    d["pos_collapsed"] = np.where(
        d["fixation_count"] <= int(keep_first_n),
        d["fixation_count"],
        int(keep_first_n) + 1,
    ).astype(int)

    trial_cols = ["subject_id", "game", "trial_number", "option"]

    # Denominator: all fixations or only relevant (non-zero reward) fixations.
    if denom == "all":
        d_total = d.copy()
    else:
        d_total = d[d["is_relevant"] == 1].copy()
        d_total[reward_col] = pd.to_numeric(d_total[reward_col], errors="coerce")
        d_total = d_total.dropna(subset=[reward_col])
        d_total = d_total[d_total[reward_col] != 0]

    # Numerator: relevant items with known non-zero reward.
    d_num = d[d["is_relevant"] == 1].copy()
    d_num[reward_col] = pd.to_numeric(d_num[reward_col], errors="coerce")
    d_num = d_num.dropna(subset=[reward_col])
    d_num = d_num[d_num[reward_col] != 0]

    if d_total.empty or d_num.empty:
        return pd.DataFrame(columns=["pos_collapsed", "valence", "prop_mean", "prop_sem"])

    d_num["valence"] = np.where(d_num[reward_col] > 0, "positive", "negative")

    # Count fixations per (trial, position, valence) — numerator.
    counts = d_num.groupby(trial_cols + ["pos_collapsed", "valence"], as_index=False).agg(
        n_fix=("valence", "size")
    )
    # Count all fixations per (trial, position) — denominator.
    totals = d_total.groupby(trial_cols + ["pos_collapsed"], as_index=False).agg(
        n_total=("pos_collapsed", "size")
    )
    totals = totals[totals["n_total"] > 0].copy()

    # Expand grid so zero-count (trial, position, valence) combinations get 0.
    vals = pd.DataFrame({"valence": ["positive", "negative"]})
    grid = totals.assign(_k=1).merge(vals.assign(_k=1), on="_k").drop(columns=["_k"])
    counts = grid.merge(counts, on=trial_cols + ["pos_collapsed", "valence"], how="left")
    counts["n_fix"] = pd.to_numeric(counts["n_fix"], errors="coerce").fillna(0.0)
    counts["prop"] = counts["n_fix"] / counts["n_total"]

    # Within-subject means.
    per_sub = counts.groupby(["subject_id", "pos_collapsed", "valence"], as_index=False).agg(
        prop=("prop", "mean")
    )

    def _sem_s(x: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce").dropna()
        if len(x) <= 1:
            return float("nan")
        return float(x.std(ddof=1) / np.sqrt(len(x)))

    summary = per_sub.groupby(["pos_collapsed", "valence"], as_index=False).agg(
        prop_mean=("prop", "mean"),
        prop_sem=("prop", _sem_s),
    )
    summary["pos_collapsed"] = pd.to_numeric(summary["pos_collapsed"], errors="coerce").astype(int)
    return summary.sort_values(["pos_collapsed", "valence"]).reset_index(drop=True)


def _compute_chance_positive_by_decision(
    fix: pd.DataFrame,
    *,
    base_dir: Path,
    excluded_subjects: Tuple[str, ...],
    decision: int,
    denom: str = "relevant",
    valence: str = "positive",
) -> float:
    """Compute chance proportion for a given valence among relevant items.

    denom='relevant': chance = (# valence-relevant items) / (# relevant items).
    denom='all': chance = (# valence-relevant items) / (# all items per trial).

    Uses logfile-derived true rewards.  Average within-subject, then across.
    """
    subjects = sorted(fix["subject_id"].dropna().astype(str).unique().tolist())
    true_rewards = load_true_rewards(
        base_dir, subjects=subjects, excluded_subjects=tuple(excluded_subjects),
    )
    if true_rewards is None or true_rewards.empty:
        return 0.5

    # Get unique (subject, game, option) trials and their choice.
    trial_cols = ["subject_id", "game", "trial_number", "option"]
    trials = fix[trial_cols + ["choice"]].drop_duplicates(subset=trial_cols).copy()
    trials["choice"] = pd.to_numeric(trials["choice"], errors="coerce")
    trials = trials[trials["choice"] == decision].copy()
    if trials.empty:
        return 0.5

    # For each (subject, game, option), find ALL items via token rule.
    opt_items = trials[["subject_id", "game", "option"]].drop_duplicates().merge(
        true_rewards, on=["subject_id", "game"], how="left",
    )
    opt_items = opt_items.dropna(subset=["image", "reward_true"]).copy()
    opt_items["image"] = opt_items["image"].astype(str)
    opt_items["option"] = opt_items["option"].astype(str)
    opt_items["is_relevant"] = [
        int(isinstance(o, str) and isinstance(i, str) and o in i.split("_"))
        for o, i in zip(opt_items["option"].tolist(), opt_items["image"].tolist())
    ]

    opt_items["reward_true"] = pd.to_numeric(opt_items["reward_true"], errors="coerce")
    if valence == "positive":
        opt_items["is_target"] = ((opt_items["is_relevant"] == 1) & (opt_items["reward_true"] > 0)).astype(int)
    else:
        opt_items["is_target"] = ((opt_items["is_relevant"] == 1) & (opt_items["reward_true"] < 0)).astype(int)

    if denom == "all":
        # chance = (# valence-relevant) / (# total items per trial)
        trial_chance = opt_items.groupby(["subject_id", "game", "option"], as_index=False).agg(
            n_target=("is_target", "sum"),
            n_total=("is_target", "size"),
        )
        trial_chance["chance"] = trial_chance["n_target"] / trial_chance["n_total"]
    else:
        # chance = (# valence-relevant) / (# relevant items per trial)
        rel = opt_items[opt_items["is_relevant"] == 1].copy()
        if rel.empty:
            return 0.5
        trial_chance = rel.groupby(["subject_id", "game", "option"], as_index=False).agg(
            n_target=("is_target", "sum"),
            n_rel=("is_target", "size"),
        )
        trial_chance["chance"] = trial_chance["n_target"] / trial_chance["n_rel"]

    per_sub = trial_chance.groupby("subject_id", as_index=False).agg(
        chance=("chance", "mean")
    )
    return float(per_sub["chance"].mean())


def _panel_prop_positive_negative_relevant_by_decision(
    ax_take: plt.Axes,
    ax_leave: plt.Axes,
    *,
    base_dir: Path,
    clean_choice_fixations_path: Optional[Path] = None,
    title: str,
    max_fixations: int,
    keep_first_n: int,
    excluded_subjects: Tuple[str, ...] = tuple(DEFAULT_EXCLUDED_EYE),
    y_lim: Tuple[float, float] = (0.2, 0.8),
    denom: str = "relevant",
    reward_col: str = "reward",
    fix_df: Optional[pd.DataFrame] = None,
    precomputed_summaries: Optional[dict] = None,
) -> None:
    """Plot prop positive/negative relevant fixations by position, split take/leave.

    ax_take: top subplot (Take trials)
    ax_leave: bottom subplot (Leave trials)
    denom: 'relevant' — among relevant fixations (sums to 1).
           'all' — among all fixations incl. irrelevant (like overview figure).
    reward_col: column to use for valence sign. 'reward' for true reward,
               'reward_recalled' for participant-reported values.
    precomputed_summaries: optional dict mapping decision (1 or 2) -> summary DataFrame.
    """
    if precomputed_summaries is None:
        if fix_df is not None:
            fix = fix_df
        else:
            fix = load_clean_choice_fixations_full(clean_choice_fixations_path)
            excluded = set(map(str, excluded_subjects))
            if excluded:
                fix = fix[~fix["subject_id"].isin(excluded)].copy()
        if fix.empty:
            raise RuntimeError("No fixations after exclusions")

        # If using recalled rewards, load and merge them into the fixation data.
        if reward_col == "reward_recalled" and "reward_recalled" not in fix.columns:
            recalled = load_recalled_rewards(base_dir, excluded_subjects=excluded_subjects)
            fix = fix.merge(
                recalled[["subject_id", "game", "image", "reward_recalled"]],
                on=["subject_id", "game", "image"],
                how="left",
            )

    colors = {"positive": "tab:blue", "negative": "tab:orange"}
    labels = {"positive": "Positive", "negative": "Negative"}
    band_alpha = 0.20

    for ax, decision, dec_label in [(ax_take, 1, "Take"), (ax_leave, 2, "Leave")]:
        if precomputed_summaries is not None:
            summary = precomputed_summaries[decision]
        else:
            summary = _compute_prop_positive_negative_relevant_by_decision(
                fix,
                max_fixations=int(max_fixations),
                keep_first_n=int(keep_first_n),
                decision=decision,
                denom=denom,
                reward_col=reward_col,
            )
        for val in ["positive", "negative"]:
            s = summary[summary["valence"] == val].copy().sort_values("pos_collapsed")
            if s.empty:
                continue
            x = s["pos_collapsed"].to_numpy(dtype=float)
            y = s["prop_mean"].to_numpy(dtype=float)
            se = s["prop_sem"].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            x, y, se = x[ok], y[ok], se[ok]

            ax.plot(x, y, color=colors[val], linewidth=2.5, label=labels[val])
            if np.isfinite(se).any() and len(se) == len(y):
                lo = np.clip(y - se, 0.0, 1.0)
                hi = np.clip(y + se, 0.0, 1.0)
                ax.fill_between(x, lo, hi, color=colors[val], alpha=band_alpha, linewidth=0)

        # Single chance baseline at 0.25 (uniform: 1.5 relevant items out of 6 total
        # per valence on average).
        ax.axhline(0.25, color="black", linewidth=1.5, linestyle=":")

        ax.set_ylim(float(y_lim[0]), float(y_lim[1]))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)

        ticks = list(range(1, int(keep_first_n) + 2))
        ax.set_xticks(ticks)
        tail_label = f"{int(keep_first_n) + 1}+"
        ax.set_xticklabels([str(i) for i in range(1, int(keep_first_n) + 1)] + [tail_label])

    # Take subplot: subtitle, hide x tick labels.
    ax_take.set_title(f"{title} — Take")
    ax_take.set_xticklabels([])

    # Leave subplot: subtitle, show fixation number labels, x-label.
    ax_leave.set_title(f"{title} — Leave", fontsize=14)
    ax_leave.set_xlabel("Fixation number")

    # Shared y-axis label on the take (top) subplot.
    ax_take.set_ylabel("Proportion of Fixations", fontsize=14)
    ax_leave.set_ylabel("")

    # Legend on top subplot only.
    ax_take.legend(frameon=False, fontsize=10, loc="best")


# ---------------------------------------------------------------------------
# Trial-level grouping helpers for sign-alignment and exact-cancellation
# ---------------------------------------------------------------------------


def _build_trial_groupings(
    *,
    base_dir: Path,
    fix: pd.DataFrame,
    excluded_subjects: Tuple[str, ...],
) -> pd.DataFrame:
    """For each trial (subject_id, game, option) compute:

    - offer_value: sum of the 3 relevant items' reward_true.
    - sign_misaligned: True when the highest |V| item's sign opposes the
      offer_value sign.  Ties at rank-1: if tied items share the same sign
      → use that sign; if mixed → classify as misaligned.
    - exact_cancel: True when the top-2 |V| items have equal |reward_true|
      AND opposite signs (e.g. -9 and 9).

    Returns a DataFrame with one row per (subject_id, game, option).
    """

    if fix.empty:
        return pd.DataFrame(
            columns=[
                "subject_id", "game", "option",
                "offer_value", "sign_misaligned", "exact_cancel",
            ]
        )

    subjects = sorted(fix["subject_id"].dropna().astype(str).unique().tolist())
    true_rewards = load_true_rewards(base_dir, subjects=subjects, excluded_subjects=tuple(excluded_subjects))
    if true_rewards is None or true_rewards.empty:
        raise RuntimeError(f"No true rewards found under {base_dir}; cannot compute trial groupings.")

    opts = fix[["subject_id", "game", "option"]].drop_duplicates().copy()
    opt_items = opts.merge(true_rewards, on=["subject_id", "game"], how="left")
    opt_items = opt_items.dropna(subset=["image", "reward_true"]).copy()
    opt_items["image"] = opt_items["image"].astype(str)
    opt_items["option"] = opt_items["option"].astype(str)
    opt_items["reward_true"] = pd.to_numeric(opt_items["reward_true"], errors="coerce")

    opt_items["is_relevant_token"] = [
        int(isinstance(opt, str) and isinstance(img, str) and opt in img.split("_"))
        for opt, img in zip(opt_items["option"].tolist(), opt_items["image"].tolist())
    ]

    rel = opt_items[opt_items["is_relevant_token"] == 1].copy()
    if rel.empty:
        raise RuntimeError("No relevant items found when computing trial groupings.")

    # Rank by |reward_true| descending within each trial.
    rel["abs_reward"] = rel["reward_true"].abs()
    rel["rank_desc"] = rel.groupby(["subject_id", "game", "option"], sort=False)["abs_reward"].rank(
        method="min", ascending=False
    )
    rel["rank_desc"] = pd.to_numeric(rel["rank_desc"], errors="coerce")

    # Per-trial aggregation.
    trial_cols = ["subject_id", "game", "option"]
    offer_values = rel.groupby(trial_cols, as_index=False).agg(offer_value=("reward_true", "sum"))

    results = []
    for (sid, g, opt), grp in rel.groupby(trial_cols, sort=False):
        ov = float(grp["reward_true"].sum())

        # --- sign_misaligned ---
        top1 = grp[grp["rank_desc"] == grp["rank_desc"].min()].copy()
        top1_signs = set(np.sign(top1["reward_true"].to_numpy(dtype=float)).astype(int))
        top1_signs.discard(0)
        if len(top1_signs) != 1:
            # Mixed signs among tied-at-rank-1 items → misaligned.
            sign_misaligned = True
        else:
            top1_sign = top1_signs.pop()
            ov_sign = int(np.sign(ov))
            sign_misaligned = top1_sign != ov_sign

        # --- exact_cancel ---
        # Identify the two highest |reward| items. If there are ties at
        # rank 1 spanning ≥2 items, the "top 2" are the tied items.
        # Otherwise, rank-1 + rank-2 items.
        sorted_by_abs = grp.sort_values("abs_reward", ascending=False).reset_index(drop=True)
        if len(sorted_by_abs) < 2:
            exact_cancel = False
        else:
            top2 = sorted_by_abs.iloc[:2]
            r1 = float(top2.iloc[0]["reward_true"])
            r2 = float(top2.iloc[1]["reward_true"])
            exact_cancel = bool(abs(r1) == abs(r2) and np.sign(r1) != np.sign(r2))

        results.append({
            "subject_id": str(sid),
            "game": int(g),
            "option": str(opt),
            "offer_value": float(ov),
            "sign_misaligned": bool(sign_misaligned),
            "exact_cancel": bool(exact_cancel),
        })

    groupings = pd.DataFrame(results)
    return groupings


def _panel_absvalue_bin_by_position_filtered(
    ax: plt.Axes,
    *,
    base_dir: Path,
    clean_choice_fixations_path: Path,
    title: str,
    max_fixations: int,
    keep_first_n: int,
    excluded_subjects: Tuple[str, ...] = tuple(DEFAULT_EXCLUDED_EYE),
    y_lim: Optional[Tuple[float, float]] = None,
    trial_filter: Optional[pd.DataFrame] = None,
) -> None:
    """Same as _panel_prop_relevant_fixations_within_absvalue_bin_by_position_7plus
    but optionally restricted to a subset of trials given by *trial_filter*
    (must contain columns: subject_id, game, option).
    """

    fix = load_clean_choice_fixations_full(clean_choice_fixations_path)
    excluded = set(map(str, excluded_subjects))
    if excluded:
        fix = fix[~fix["subject_id"].isin(excluded)].copy()
    if fix.empty:
        raise RuntimeError(f"No fixations left after exclusions for: {clean_choice_fixations_path}")

    # Apply trial filter before any computation.
    if trial_filter is not None and not trial_filter.empty:
        tf = trial_filter[["subject_id", "game", "option"]].drop_duplicates().copy()
        tf["subject_id"] = tf["subject_id"].astype(str)
        tf["game"] = pd.to_numeric(tf["game"], errors="coerce").astype(int)
        tf["option"] = tf["option"].astype(str)
        fix = fix.merge(tf, on=["subject_id", "game", "option"], how="inner")
    if fix.empty:
        ax.set_title(title + " (no trials)")
        ax.spines[["top", "right"]].set_visible(False)
        return

    rank_map, chance = _build_relevant_item_rank_map_and_chance(
        base_dir=base_dir,
        fix=fix,
        excluded_subjects=tuple(excluded_subjects),
    )

    d = fix.copy()
    d["fixation_count"] = pd.to_numeric(d["fixation_count"], errors="coerce")
    d = d.dropna(subset=["fixation_count"]).copy()
    d["fixation_count"] = d["fixation_count"].astype(int)
    d = d[d["fixation_count"] <= int(max_fixations)].copy()
    d = d[d["is_relevant"] == 1].copy()
    if d.empty:
        ax.set_title(title + " (no relevant fixations)")
        ax.spines[["top", "right"]].set_visible(False)
        return

    d = d.merge(rank_map, on=["subject_id", "game", "option", "image"], how="left")
    d["rank_bin"] = pd.to_numeric(d["rank_bin"], errors="coerce")
    d = d.dropna(subset=["rank_bin"]).copy()
    d["rank_bin"] = d["rank_bin"].astype(int)
    d = d[d["rank_bin"].isin([1, 2, 3])].copy()

    trial_cols = ["subject_id", "game", "trial_number", "option"]
    d["fixation_position"] = d["fixation_count"].astype(int)
    d["pos_collapsed"] = np.where(
        d["fixation_position"] <= int(keep_first_n),
        d["fixation_position"],
        int(keep_first_n) + 1,
    ).astype(int)

    counts = d.groupby(trial_cols + ["pos_collapsed", "rank_bin"], as_index=False).agg(n_fix=("rank_bin", "size"))
    totals = d.groupby(trial_cols + ["pos_collapsed"], as_index=False).agg(n_total=("rank_bin", "size"))
    totals = totals[totals["n_total"] > 0].copy()

    bins = pd.DataFrame({"rank_bin": [1, 2, 3]})
    grid = totals.assign(_k=1).merge(bins.assign(_k=1), on="_k", how="left").drop(columns=["_k"])
    counts = grid.merge(counts, on=trial_cols + ["pos_collapsed", "rank_bin"], how="left")
    counts["n_fix"] = pd.to_numeric(counts["n_fix"], errors="coerce").fillna(0.0)
    counts["prop"] = counts["n_fix"] / counts["n_total"]

    per_sub = counts.groupby(["subject_id", "pos_collapsed", "rank_bin"], as_index=False).agg(prop=("prop", "mean"))

    def _sem_series(x: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce").dropna()
        if len(x) <= 1:
            return float("nan")
        return float(x.std(ddof=1) / np.sqrt(len(x)))

    summary = per_sub.groupby(["pos_collapsed", "rank_bin"], as_index=False).agg(
        prop_mean=("prop", "mean"),
        prop_sem=("prop", _sem_series),
        n_subjects=("subject_id", lambda s: int(pd.Series(s).nunique())),
    )
    summary["pos_collapsed"] = pd.to_numeric(summary["pos_collapsed"], errors="coerce").astype(int)
    summary["rank_bin"] = pd.to_numeric(summary["rank_bin"], errors="coerce").astype(int)

    # Chance baselines (restricted to the same trial subset via the filtered fix).
    trials = fix[trial_cols].drop_duplicates().copy()
    chance_t = trials.merge(chance, on=["subject_id", "game", "option"], how="left")
    chance_long = chance_t.melt(
        id_vars=trial_cols,
        value_vars=["chance_prop_bin1", "chance_prop_bin2", "chance_prop_bin3"],
        var_name="chance_bin",
        value_name="chance_prop",
    )
    chance_long["rank_bin"] = chance_long["chance_bin"].map(
        {"chance_prop_bin1": 1, "chance_prop_bin2": 2, "chance_prop_bin3": 3}
    )
    chance_long["chance_prop"] = pd.to_numeric(chance_long["chance_prop"], errors="coerce")
    chance_sub = chance_long.groupby(["subject_id", "rank_bin"], as_index=False).agg(chance_prop=("chance_prop", "mean"))
    chance_sum = chance_sub.groupby("rank_bin", as_index=False).agg(
        chance_mean=("chance_prop", "mean"),
        chance_sem=("chance_prop", _sem_series),
    )

    # Plotting (same style as G/H — purple palette).
    colors = {1: "#54278f", 2: "#756bb1", 3: "#9e9ac8"}
    labels = {1: "High |V|", 2: "Med. |V|", 3: "Low |V|"}
    band_alpha = 0.20

    for b in [1, 2, 3]:
        s = summary[summary["rank_bin"] == b].copy().sort_values("pos_collapsed")
        if s.empty:
            continue
        x = pd.to_numeric(s["pos_collapsed"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(s["prop_mean"], errors="coerce").to_numpy(dtype=float)
        se = pd.to_numeric(s["prop_sem"], errors="coerce").to_numpy(dtype=float)

        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]
        y = y[ok]
        se = se[ok] if len(se) == len(ok) else se

        ax.plot(x, y, color=colors[b], linewidth=2.5, label=labels[b])
        if np.isfinite(se).any() and len(se) == len(y):
            lo = np.clip(y - se, 0.0, 1.0)
            hi = np.clip(y + se, 0.0, 1.0)
            ax.fill_between(x, lo, hi, color=colors[b], alpha=band_alpha, linewidth=0)

        c = chance_sum[chance_sum["rank_bin"] == b]
        if not c.empty:
            chance_y = float(pd.to_numeric(c["chance_mean"].iloc[0], errors="coerce"))
            if np.isfinite(chance_y):
                ax.axhline(chance_y, color=colors[b], linewidth=1.5, linestyle=":")

    ax.set_title(title)
    ax.set_xlabel("Fixation number")
    ax.set_ylabel("Prop. relevant fixated")
    if y_lim is not None:
        ax.set_ylim(float(y_lim[0]), float(y_lim[1]))
    else:
        # Auto-scale with a small margin.
        cur_lo, cur_hi = ax.get_ylim()
        margin = (cur_hi - cur_lo) * 0.08
        ax.set_ylim(max(0.0, cur_lo - margin), min(1.0, cur_hi + margin))

    ticks = list(range(1, int(keep_first_n) + 2))
    ax.set_xticks(ticks)
    tail_label = f"{int(keep_first_n) + 1}+"
    ax.set_xticklabels([str(i) for i in range(1, int(keep_first_n) + 1)] + [tail_label])
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="best")


def _panel_trial_group_proportion(
    ax: plt.Axes,
    *,
    groupings: pd.DataFrame,
    col: str,
    title: str,
    y_label: str = "Proportion of trials",
    bar_color: str = "0.4",
) -> None:
    """Plot per-subject mean ± SEM of the proportion of trials where *col* is True."""

    g = groupings.copy()
    g[col] = g[col].astype(bool).astype(int)

    per_sub = g.groupby("subject_id", as_index=False).agg(prop=(col, "mean"))
    vals = per_sub["prop"].to_numpy(dtype=float)

    mean_val = float(np.nanmean(vals))
    sem_val = _sem(vals)

    ax.bar([0], [mean_val], width=0.5, color=bar_color, alpha=0.85, edgecolor="none")
    ax.errorbar([0], [mean_val], yerr=[sem_val], fmt="none", ecolor="k", capsize=5, linewidth=1.2)

    rng = np.random.default_rng(42)
    jit = rng.uniform(-0.12, 0.12, size=len(vals))
    ax.scatter(jit, vals, s=28, color="0.15", alpha=0.50, edgecolor="none", zorder=3)

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate mean.
    ax.text(
        0.0, mean_val + sem_val + 0.03, f"{mean_val:.2f}",
        ha="center", va="bottom", fontsize=14, fontweight="bold",
    )


def _save_grouped_absvalue_figure(
    *,
    out_dir: Path,
    tag: str,
    figure_name: str,
    grouping_col: str,
    condition_true_label: str,
    condition_false_label: str,
    proportion_title: str,
    human_base_dir: Path,
    nn_base_dir: Path,
    human_clean_path: Path,
    nn_clean_path: Path,
    human_groupings: pd.DataFrame,
    nn_groupings: pd.DataFrame,
    max_fixations: int,
    keep_first_n: int,
    excluded_subjects: Tuple[str, ...] = tuple(DEFAULT_EXCLUDED_EYE),
    y_lim: Optional[Tuple[float, float]] = None,
) -> Path:
    """Create a 2×3 figure.

    Row 1 = Humans, Row 2 = Network.
    Col 1 = trials where grouping_col is True.
    Col 2 = trials where grouping_col is False.
    Col 3 = proportion of trials in the True condition.
    """

    _ensure_dir(out_dir)
    suffix = f"_{tag}" if tag else ""
    out_path = out_dir / f"{figure_name}{suffix}.pdf"

    # Split trial filters.
    h_true = human_groupings[human_groupings[grouping_col] == True][["subject_id", "game", "option"]]  # noqa: E712
    h_false = human_groupings[human_groupings[grouping_col] == False][["subject_id", "game", "option"]]  # noqa: E712
    n_true = nn_groupings[nn_groupings[grouping_col] == True][["subject_id", "game", "option"]]  # noqa: E712
    n_false = nn_groupings[nn_groupings[grouping_col] == False][["subject_id", "game", "option"]]  # noqa: E712

    # Print sanity counts.
    h_total = len(human_groupings)
    n_total = len(nn_groupings)
    print(f"\n[{figure_name}] Trial counts:")
    print(f"  Human  — {grouping_col}=True: {len(h_true)}, False: {len(h_false)}, Total: {h_total}")
    print(f"  Network — {grouping_col}=True: {len(n_true)}, False: {len(n_false)}, Total: {n_total}")

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["xtick.major.width"] = 2
    plt.rcParams["ytick.major.width"] = 2

    with plt.rc_context({"axes.labelsize": 18, "axes.titlesize": 18, "xtick.labelsize": 14, "ytick.labelsize": 14}):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Row 1: Humans.
        _panel_absvalue_bin_by_position_filtered(
            axes[0, 0],
            base_dir=human_base_dir,
            clean_choice_fixations_path=human_clean_path,
            title=f"Humans — {condition_true_label}",
            max_fixations=max_fixations,
            keep_first_n=keep_first_n,
            excluded_subjects=excluded_subjects,
            y_lim=y_lim,
            trial_filter=h_true,
        )
        _add_panel_label(axes[0, 0], "A", dx=-75)

        _panel_absvalue_bin_by_position_filtered(
            axes[0, 1],
            base_dir=human_base_dir,
            clean_choice_fixations_path=human_clean_path,
            title=f"Humans — {condition_false_label}",
            max_fixations=max_fixations,
            keep_first_n=keep_first_n,
            excluded_subjects=excluded_subjects,
            y_lim=y_lim,
            trial_filter=h_false,
        )
        _add_panel_label(axes[0, 1], "B", dx=-75)

        _panel_trial_group_proportion(
            axes[0, 2],
            groupings=human_groupings,
            col=grouping_col,
            title=f"Humans — {proportion_title}",
        )
        _add_panel_label(axes[0, 2], "C", dx=-75)

        # Row 2: Network.
        _panel_absvalue_bin_by_position_filtered(
            axes[1, 0],
            base_dir=nn_base_dir,
            clean_choice_fixations_path=nn_clean_path,
            title=f"Network — {condition_true_label}",
            max_fixations=max_fixations,
            keep_first_n=keep_first_n,
            excluded_subjects=excluded_subjects,
            y_lim=y_lim,
            trial_filter=n_true,
        )
        _add_panel_label(axes[1, 0], "D", dx=-75)

        _panel_absvalue_bin_by_position_filtered(
            axes[1, 1],
            base_dir=nn_base_dir,
            clean_choice_fixations_path=nn_clean_path,
            title=f"Network — {condition_false_label}",
            max_fixations=max_fixations,
            keep_first_n=keep_first_n,
            excluded_subjects=excluded_subjects,
            y_lim=y_lim,
            trial_filter=n_false,
        )
        _add_panel_label(axes[1, 1], "E", dx=-75)

        _panel_trial_group_proportion(
            axes[1, 2],
            groupings=nn_groupings,
            col=grouping_col,
            title=f"Network — {proportion_title}",
        )
        _add_panel_label(axes[1, 2], "F", dx=-75)

        fig.tight_layout(w_pad=3.0, h_pad=3.0)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

    print(f"  Wrote {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Value-ranked (not |value|) proportion of relevant fixations by take/leave
# ---------------------------------------------------------------------------


def _rank_value_bins_1_highest(values: pd.Series) -> pd.Series:
    """Return rank bins 1..3 where 1 is the highest *signed* value.

    E.g. for rewards (9, 1, -9): 9→1, 1→2, -9→3.
    Tie rule: min rank (ascending=False).
    """
    v = pd.to_numeric(values, errors="coerce")
    r = v.rank(method="min", ascending=False)
    return r.clip(lower=1, upper=3)


def _build_relevant_item_value_rank_map_and_chance(
    *,
    base_dir: Path,
    fix: pd.DataFrame,
    excluded_subjects: Tuple[str, ...],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Like _build_relevant_item_rank_map_and_chance but ranks by signed
    reward (highest value → rank 1) instead of |reward|.

    Returns
    -------
    rank_map : (subject_id, game, option, image) -> value_rank_bin (1..3)
    chance   : (subject_id, game, option) -> chance_prop_bin{1,2,3}
    """

    if fix.empty:
        return (
            pd.DataFrame(columns=["subject_id", "game", "option", "image", "value_rank_bin"]),
            pd.DataFrame(columns=["subject_id", "game", "option",
                                   "chance_prop_bin1", "chance_prop_bin2", "chance_prop_bin3"]),
        )

    subjects = sorted(fix["subject_id"].dropna().astype(str).unique().tolist())
    true_rewards = load_true_rewards(base_dir, subjects=subjects,
                                     excluded_subjects=tuple(excluded_subjects))
    if true_rewards is None or true_rewards.empty:
        raise RuntimeError(f"No true rewards found under {base_dir}; cannot compute value bins.")

    opts = fix[["subject_id", "game", "option"]].drop_duplicates().copy()
    opt_items = opts.merge(true_rewards, on=["subject_id", "game"], how="left")
    opt_items = opt_items.dropna(subset=["image", "reward_true"]).copy()
    opt_items["image"] = opt_items["image"].astype(str)
    opt_items["option"] = opt_items["option"].astype(str)

    opt_items["is_relevant_token"] = [
        int(isinstance(opt, str) and isinstance(img, str) and opt in img.split("_"))
        for opt, img in zip(opt_items["option"].tolist(), opt_items["image"].tolist())
    ]

    rel = opt_items[opt_items["is_relevant_token"] == 1].copy()
    if rel.empty:
        raise RuntimeError("No relevant items found when computing value bins.")

    rel["value_rank_bin"] = rel.groupby(
        ["subject_id", "game", "option"], sort=False
    )["reward_true"].transform(_rank_value_bins_1_highest)
    rel["value_rank_bin"] = pd.to_numeric(rel["value_rank_bin"], errors="coerce").astype("Int64")
    rel = rel.dropna(subset=["value_rank_bin"]).copy()
    rel["value_rank_bin"] = rel["value_rank_bin"].astype(int).clip(lower=1, upper=3)

    rank_map = rel[["subject_id", "game", "option", "image", "value_rank_bin"]].drop_duplicates(
        subset=["subject_id", "game", "option", "image"]
    )

    # Chance proportions.
    counts = rank_map.groupby(
        ["subject_id", "game", "option", "value_rank_bin"], as_index=False
    ).agg(n_items=("image", "size"))
    chance = (
        counts.pivot_table(
            index=["subject_id", "game", "option"],
            columns="value_rank_bin",
            values="n_items",
            fill_value=0,
        )
        .reset_index()
        .rename(columns={1: "n_bin1", 2: "n_bin2", 3: "n_bin3"})
    )
    for col in ["n_bin1", "n_bin2", "n_bin3"]:
        if col not in chance.columns:
            chance[col] = 0
    chance["chance_prop_bin1"] = pd.to_numeric(chance["n_bin1"], errors="coerce").fillna(0.0) / 3.0
    chance["chance_prop_bin2"] = pd.to_numeric(chance["n_bin2"], errors="coerce").fillna(0.0) / 3.0
    chance["chance_prop_bin3"] = pd.to_numeric(chance["n_bin3"], errors="coerce").fillna(0.0) / 3.0
    chance = chance[["subject_id", "game", "option",
                     "chance_prop_bin1", "chance_prop_bin2", "chance_prop_bin3"]]

    return rank_map, chance


def _panel_value_rank_by_position_filtered(
    ax: plt.Axes,
    *,
    base_dir: Path,
    clean_choice_fixations_path: Path,
    title: str,
    max_fixations: int,
    keep_first_n: int,
    excluded_subjects: Tuple[str, ...] = tuple(DEFAULT_EXCLUDED_EYE),
    y_lim: Optional[Tuple[float, float]] = None,
    choice_filter: Optional[int] = None,
) -> None:
    """Plot proportion of relevant fixations in each signed-value rank bin
    by fixation position (7+ collapsed).

    Parameters
    ----------
    choice_filter : 1 = take trials only, 2 = leave trials only, None = all.
    """

    fix = load_clean_choice_fixations_full(clean_choice_fixations_path)
    excluded = set(map(str, excluded_subjects))
    if excluded:
        fix = fix[~fix["subject_id"].isin(excluded)].copy()
    if fix.empty:
        raise RuntimeError(f"No fixations left after exclusions for: {clean_choice_fixations_path}")

    # Filter by choice (take/leave).
    if choice_filter is not None:
        fix["choice"] = pd.to_numeric(fix["choice"], errors="coerce")
        fix = fix[fix["choice"] == int(choice_filter)].copy()
    if fix.empty:
        ax.set_title(title + " (no trials)")
        ax.spines[["top", "right"]].set_visible(False)
        return

    rank_map, chance = _build_relevant_item_value_rank_map_and_chance(
        base_dir=base_dir,
        fix=fix,
        excluded_subjects=tuple(excluded_subjects),
    )

    d = fix.copy()
    d["fixation_count"] = pd.to_numeric(d["fixation_count"], errors="coerce")
    d = d.dropna(subset=["fixation_count"]).copy()
    d["fixation_count"] = d["fixation_count"].astype(int)
    d = d[d["fixation_count"] <= int(max_fixations)].copy()
    d = d[d["is_relevant"] == 1].copy()
    if d.empty:
        ax.set_title(title + " (no relevant fixations)")
        ax.spines[["top", "right"]].set_visible(False)
        return

    d = d.merge(rank_map, on=["subject_id", "game", "option", "image"], how="left")
    d["value_rank_bin"] = pd.to_numeric(d["value_rank_bin"], errors="coerce")
    d = d.dropna(subset=["value_rank_bin"]).copy()
    d["value_rank_bin"] = d["value_rank_bin"].astype(int)
    d = d[d["value_rank_bin"].isin([1, 2, 3])].copy()

    trial_cols = ["subject_id", "game", "trial_number", "option"]
    d["fixation_position"] = d["fixation_count"].astype(int)
    d["pos_collapsed"] = np.where(
        d["fixation_position"] <= int(keep_first_n),
        d["fixation_position"],
        int(keep_first_n) + 1,
    ).astype(int)

    counts = d.groupby(trial_cols + ["pos_collapsed", "value_rank_bin"], as_index=False).agg(
        n_fix=("value_rank_bin", "size")
    )
    totals = d.groupby(trial_cols + ["pos_collapsed"], as_index=False).agg(
        n_total=("value_rank_bin", "size")
    )
    totals = totals[totals["n_total"] > 0].copy()

    bins = pd.DataFrame({"value_rank_bin": [1, 2, 3]})
    grid = totals.assign(_k=1).merge(bins.assign(_k=1), on="_k", how="left").drop(columns=["_k"])
    counts = grid.merge(counts, on=trial_cols + ["pos_collapsed", "value_rank_bin"], how="left")
    counts["n_fix"] = pd.to_numeric(counts["n_fix"], errors="coerce").fillna(0.0)
    counts["prop"] = counts["n_fix"] / counts["n_total"]

    per_sub = counts.groupby(
        ["subject_id", "pos_collapsed", "value_rank_bin"], as_index=False
    ).agg(prop=("prop", "mean"))

    def _sem_series(x: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce").dropna()
        if len(x) <= 1:
            return float("nan")
        return float(x.std(ddof=1) / np.sqrt(len(x)))

    summary = per_sub.groupby(["pos_collapsed", "value_rank_bin"], as_index=False).agg(
        prop_mean=("prop", "mean"),
        prop_sem=("prop", _sem_series),
        n_subjects=("subject_id", lambda s: int(pd.Series(s).nunique())),
    )
    summary["pos_collapsed"] = pd.to_numeric(summary["pos_collapsed"], errors="coerce").astype(int)
    summary["value_rank_bin"] = pd.to_numeric(summary["value_rank_bin"], errors="coerce").astype(int)

    # Chance baselines.
    trials = fix[trial_cols].drop_duplicates().copy()
    chance_t = trials.merge(chance, on=["subject_id", "game", "option"], how="left")
    chance_long = chance_t.melt(
        id_vars=trial_cols,
        value_vars=["chance_prop_bin1", "chance_prop_bin2", "chance_prop_bin3"],
        var_name="chance_bin",
        value_name="chance_prop",
    )
    chance_long["value_rank_bin"] = chance_long["chance_bin"].map(
        {"chance_prop_bin1": 1, "chance_prop_bin2": 2, "chance_prop_bin3": 3}
    )
    chance_long["chance_prop"] = pd.to_numeric(chance_long["chance_prop"], errors="coerce")
    chance_sub = chance_long.groupby(
        ["subject_id", "value_rank_bin"], as_index=False
    ).agg(chance_prop=("chance_prop", "mean"))
    chance_sum = chance_sub.groupby("value_rank_bin", as_index=False).agg(
        chance_mean=("chance_prop", "mean"),
        chance_sem=("chance_prop", _sem_series),
    )

    # Plotting — purple palette, same as G/H.
    colors = {1: "#54278f", 2: "#756bb1", 3: "#9e9ac8"}
    labels = {1: "High V", 2: "Med. V", 3: "Low V"}
    band_alpha = 0.20

    for b in [1, 2, 3]:
        s = summary[summary["value_rank_bin"] == b].copy().sort_values("pos_collapsed")
        if s.empty:
            continue
        x = pd.to_numeric(s["pos_collapsed"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(s["prop_mean"], errors="coerce").to_numpy(dtype=float)
        se = pd.to_numeric(s["prop_sem"], errors="coerce").to_numpy(dtype=float)

        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]
        y = y[ok]
        se = se[ok] if len(se) == len(ok) else se

        ax.plot(x, y, color=colors[b], linewidth=2.5, label=labels[b])
        if np.isfinite(se).any() and len(se) == len(y):
            lo = np.clip(y - se, 0.0, 1.0)
            hi = np.clip(y + se, 0.0, 1.0)
            ax.fill_between(x, lo, hi, color=colors[b], alpha=band_alpha, linewidth=0)

        c = chance_sum[chance_sum["value_rank_bin"] == b]
        if not c.empty:
            chance_y = float(pd.to_numeric(c["chance_mean"].iloc[0], errors="coerce"))
            if np.isfinite(chance_y):
                ax.axhline(chance_y, color=colors[b], linewidth=1.5, linestyle=":")

    ax.set_title(title)
    ax.set_xlabel("Fixation number")
    ax.set_ylabel("Prop. relevant fixated")
    if y_lim is not None:
        ax.set_ylim(float(y_lim[0]), float(y_lim[1]))
    else:
        cur_lo, cur_hi = ax.get_ylim()
        margin = (cur_hi - cur_lo) * 0.08
        ax.set_ylim(max(0.0, cur_lo - margin), min(1.0, cur_hi + margin))

    ticks = list(range(1, int(keep_first_n) + 2))
    ax.set_xticks(ticks)
    tail_label = f"{int(keep_first_n) + 1}+"
    ax.set_xticklabels([str(i) for i in range(1, int(keep_first_n) + 1)] + [tail_label])
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="best")


def _save_value_rank_by_choice_figure(
    *,
    out_dir: Path,
    tag: str,
    human_base_dir: Path,
    nn_base_dir: Path,
    human_clean_path: Path,
    nn_clean_path: Path,
    max_fixations: int,
    keep_first_n: int,
    excluded_subjects: Tuple[str, ...] = tuple(DEFAULT_EXCLUDED_EYE),
) -> Path:
    """2×2 figure: rows = Humans / Network, cols = Take / Leave.

    Each panel shows proportion of relevant fixations in each signed-value
    rank bin (High V / Med. V / Low V) by fixation position.
    """

    _ensure_dir(out_dir)
    suffix = f"_{tag}" if tag else ""
    out_path = out_dir / f"FigureNNH_value_rank_by_choice{suffix}.pdf"

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["xtick.major.width"] = 2
    plt.rcParams["ytick.major.width"] = 2

    with plt.rc_context({"axes.labelsize": 18, "axes.titlesize": 18,
                         "xtick.labelsize": 14, "ytick.labelsize": 14}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Row 1: Humans.
        _panel_value_rank_by_position_filtered(
            axes[0, 0],
            base_dir=human_base_dir,
            clean_choice_fixations_path=human_clean_path,
            title="Humans — Take",
            max_fixations=max_fixations,
            keep_first_n=keep_first_n,
            excluded_subjects=excluded_subjects,
            choice_filter=1,
        )
        _add_panel_label(axes[0, 0], "A", dx=-75)

        _panel_value_rank_by_position_filtered(
            axes[0, 1],
            base_dir=human_base_dir,
            clean_choice_fixations_path=human_clean_path,
            title="Humans — Leave",
            max_fixations=max_fixations,
            keep_first_n=keep_first_n,
            excluded_subjects=excluded_subjects,
            choice_filter=2,
        )
        _add_panel_label(axes[0, 1], "B", dx=-75)

        # Row 2: Network.
        _panel_value_rank_by_position_filtered(
            axes[1, 0],
            base_dir=nn_base_dir,
            clean_choice_fixations_path=nn_clean_path,
            title="Network — Take",
            max_fixations=max_fixations,
            keep_first_n=keep_first_n,
            excluded_subjects=excluded_subjects,
            choice_filter=1,
        )
        _add_panel_label(axes[1, 0], "C", dx=-75)

        _panel_value_rank_by_position_filtered(
            axes[1, 1],
            base_dir=nn_base_dir,
            clean_choice_fixations_path=nn_clean_path,
            title="Network — Leave",
            max_fixations=max_fixations,
            keep_first_n=keep_first_n,
            excluded_subjects=excluded_subjects,
            choice_filter=2,
        )
        _add_panel_label(axes[1, 1], "D", dx=-75)

        fig.tight_layout(w_pad=3.0, h_pad=3.0)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

    print(f"  Wrote {out_path}")
    return out_path


def _trial_keys() -> list[str]:
    return ["subject_id", "game", "trial_number", "option"]


def _subject_cumtime_curve(
    df: pd.DataFrame, *, max_fixations: int, mode: str = "all_trials"
) -> np.ndarray:
    keys = _trial_keys()
    d = df.sort_values(keys + ["fixation_count"], kind="mergesort").copy()

    total = d.groupby(keys, as_index=False).agg(total_fix_time=("fixation_duration", "sum"))
    d = d.merge(total, how="left", on=keys)
    d = d[d["total_fix_time"] > 0].copy()
    if len(d) == 0:
        raise ValueError("No fixation rows after filtering.")

    d["cum_fix_time"] = d.groupby(keys)["fixation_duration"].cumsum()
    d["cum_prop_time"] = d["cum_fix_time"] / d["total_fix_time"]

    mode = str(mode).strip().lower()
    if mode not in {"all_trials", "conditional"}:
        raise ValueError("mode must be one of: all_trials, conditional")

    agg = (
        d.groupby("fixation_count", as_index=False)
        .agg(sum_cum_prop=("cum_prop_time", "sum"), n_present=("cum_prop_time", "size"))
        .sort_values("fixation_count")
        .reset_index(drop=True)
    )

    n_total_trials = int(d[keys].drop_duplicates().shape[0])
    if n_total_trials <= 0:
        raise ValueError("No trials found.")

    xs = np.arange(1, max_fixations + 1)
    out = np.full_like(xs, fill_value=np.nan, dtype=float)

    # Map present indices
    present = {int(r.fixation_count): (float(r.sum_cum_prop), int(r.n_present)) for r in agg.itertuples(index=False)}

    for i in xs:
        if i in present:
            sum_cum_prop, n_present = present[i]
        else:
            sum_cum_prop, n_present = (0.0, 0)

        if mode == "conditional":
            out[i - 1] = (sum_cum_prop / float(n_present)) if n_present > 0 else np.nan
        else:
            ended_before = n_total_trials - int(n_present)
            out[i - 1] = (sum_cum_prop + float(ended_before)) / float(n_total_trials)

    return out


def _panel_A(ax: plt.Axes, human_by_sub: pd.DataFrame, nn_by_sub: pd.DataFrame) -> None:
    # Values
    hf = pd.to_numeric(human_by_sub["firstfix_prop_time_relevant"], errors="coerce").to_numpy(dtype=float)
    hr = pd.to_numeric(human_by_sub["revisit_prop_time_relevant"], errors="coerce").to_numpy(dtype=float)
    nf = pd.to_numeric(nn_by_sub["firstfix_prop_time_relevant"], errors="coerce").to_numpy(dtype=float)
    nr = pd.to_numeric(nn_by_sub["revisit_prop_time_relevant"], errors="coerce").to_numpy(dtype=float)

    groups = ["Humans", "Network"]
    x = np.arange(2)
    w = 0.35

    # Bar means/SEMs
    means_first = [float(np.nanmean(hf)), float(np.nanmean(nf))]
    means_rev = [float(np.nanmean(hr)), float(np.nanmean(nr))]
    sem_first = [_sem(hf), _sem(nf)]
    sem_rev = [_sem(hr), _sem(nr)]

    c_first = "0.1"
    c_rev = "0.6"

    ax.bar(x - w / 2, means_first, width=w, color=c_first, alpha=0.85, label="Initial")
    ax.bar(x + w / 2, means_rev, width=w, color=c_rev, alpha=0.85, label="Revisit")

    ax.errorbar(x - w / 2, means_first, yerr=sem_first, fmt="none", ecolor="k", capsize=4, linewidth=1)
    ax.errorbar(x + w / 2, means_rev, yerr=sem_rev, fmt="none", ecolor="k", capsize=4, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Prop. relevant duration")

    # Chance line (relevant vs irrelevant items are balanced 3 vs 3).
    ax.axhline(0.5, color="black", linewidth=1.5, linestyle=":")

    # Paired overlays: connect first vs revisit within each group.
    def _overlay(vals0: np.ndarray, vals1: np.ndarray, group_index: int) -> None:
        v0 = np.asarray(vals0, dtype=float)
        v1 = np.asarray(vals1, dtype=float)
        ok = np.isfinite(v0) & np.isfinite(v1)
        v0 = v0[ok]
        v1 = v1[ok]
        if len(v0) == 0:
            return
        rng = np.random.default_rng(0)
        jit = rng.uniform(-0.04, 0.04, size=len(v0))
        x0 = float(group_index) - w / 2 + jit
        x1 = float(group_index) + w / 2 + jit
        for i in range(len(v0)):
            ax.plot([x0[i], x1[i]], [v0[i], v1[i]], color="0.4", alpha=0.35, linewidth=0.8, zorder=2)
        ax.scatter(x0, v0, s=18, color=c_first, alpha=0.55, edgecolor="none", zorder=3)
        ax.scatter(x1, v1, s=18, color=c_rev, alpha=0.55, edgecolor="none", zorder=3)

    _overlay(hf, hr, 0)
    _overlay(nf, nr, 1)

    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=12, loc="best")


def _panel_B(ax: plt.Axes, human_by_sub: pd.DataFrame, nn_by_sub: pd.DataFrame) -> None:
    # Build the four x-groups.
    groups = [
        ("Relevant", "Human"),
        ("Relevant", "Network"),
        ("Irrelevant", "Human"),
        ("Irrelevant", "Network"),
    ]
    x = np.arange(len(groups))
    w = 0.35

    c_first = "#1f77b4"
    c_rev = "#ff7f0e"

    def _vals(df: pd.DataFrame, rel: str) -> Tuple[np.ndarray, np.ndarray]:
        rel = str(rel).lower()
        if rel == "relevant":
            v_first = pd.to_numeric(df["firstfix_count_relevant_per_trial"], errors="coerce").to_numpy(dtype=float)
            v_rev = pd.to_numeric(df["revisit_count_relevant_per_trial"], errors="coerce").to_numpy(dtype=float)
        else:
            v_first = pd.to_numeric(df["firstfix_count_irrelevant_per_trial"], errors="coerce").to_numpy(dtype=float)
            v_rev = pd.to_numeric(df["revisit_count_irrelevant_per_trial"], errors="coerce").to_numpy(dtype=float)
        return v_first, v_rev

    # Means / SEMs in group order
    means_first = []
    means_rev = []
    sem_first = []
    sem_rev = []
    per_group_vals = []
    for rel, agent in groups:
        df = human_by_sub if agent == "Human" else nn_by_sub
        v_first, v_rev = _vals(df, rel)
        means_first.append(float(np.nanmean(v_first)))
        means_rev.append(float(np.nanmean(v_rev)))
        sem_first.append(_sem(v_first))
        sem_rev.append(_sem(v_rev))
        per_group_vals.append((v_first, v_rev))

    ax.bar(x - w / 2, means_first, width=w, color=c_first, alpha=0.85, label="First-fix")
    ax.bar(x + w / 2, means_rev, width=w, color=c_rev, alpha=0.85, label="Revisit")

    ax.errorbar(x - w / 2, means_first, yerr=sem_first, fmt="none", ecolor="k", capsize=4, linewidth=1)
    ax.errorbar(x + w / 2, means_rev, yerr=sem_rev, fmt="none", ecolor="k", capsize=4, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{rel}, {agent}" for (rel, agent) in groups], rotation=20, ha="right")
    ax.set_ylabel("Fixations per trial")

    # Paired overlays: connect first vs revisit within each x-group.
    for gi, (v_first, v_rev) in enumerate(per_group_vals):
        v0 = np.asarray(v_first, dtype=float)
        v1 = np.asarray(v_rev, dtype=float)
        ok = np.isfinite(v0) & np.isfinite(v1)
        v0 = v0[ok]
        v1 = v1[ok]
        if len(v0) == 0:
            continue
        rng = np.random.default_rng(0)
        jit = rng.uniform(-0.04, 0.04, size=len(v0))
        x0 = float(gi) - w / 2 + jit
        x1 = float(gi) + w / 2 + jit
        for i in range(len(v0)):
            ax.plot([x0[i], x1[i]], [v0[i], v1[i]], color="0.4", alpha=0.35, linewidth=0.8, zorder=2)
        ax.scatter(x0, v0, s=16, color=c_first, alpha=0.55, edgecolor="none", zorder=3)
        ax.scatter(x1, v1, s=16, color=c_rev, alpha=0.55, edgecolor="none", zorder=3)

    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=12, loc="best")


def _panel_C(
    ax: plt.Axes,
    *,
    human_fix: pd.DataFrame,
    nn_fix: pd.DataFrame,
    max_fixations: int,
    cumtime_mode: str,
) -> None:
    def _curves(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        curves = []
        for sid, dsub in df.groupby("subject_id", sort=True):
            try:
                v = _subject_cumtime_curve(dsub, max_fixations=max_fixations, mode=cumtime_mode)
            except Exception:
                continue
            if np.all(~np.isfinite(v)):
                continue
            curves.append(v)
        if len(curves) == 0:
            raise ValueError("No per-subject cumulative curves could be computed.")
        mat = np.vstack(curves)
        mean = np.nanmean(mat, axis=0)
        sem = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(mat.shape[0]) if mat.shape[0] > 1 else np.full_like(mean, np.nan)
        xs = np.arange(1, max_fixations + 1)
        return xs, mean, sem

    xs_h, mean_h, sem_h = _curves(human_fix)
    xs_n, mean_n, sem_n = _curves(nn_fix)

    color_h = "#1f77b4"
    color_n = "#ff7f0e"

    ax.plot(xs_h, mean_h, color=color_h, linewidth=2.5, label="Humans")
    ax.fill_between(xs_h, np.clip(mean_h - sem_h, 0, 1), np.clip(mean_h + sem_h, 0, 1), color=color_h, alpha=0.2, linewidth=0)

    ax.plot(xs_n, mean_n, color=color_n, linewidth=2.5, label="Network")
    ax.fill_between(xs_n, np.clip(mean_n - sem_n, 0, 1), np.clip(mean_n + sem_n, 0, 1), color=color_n, alpha=0.2, linewidth=0)

    ax.set_xlabel("Fixation number")
    ax.set_ylabel("Cumulative fixation-time share")
    ax.set_ylim(0.0, 1.01)
    ax.set_xlim(0.5, max_fixations + 0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=12, loc="best")


def _panel_B_split_cumtime(
    ax_h: plt.Axes,
    ax_n: plt.Axes,
    *,
    human_fix: pd.DataFrame,
    nn_fix: pd.DataFrame,
    max_fixations: int,
    cumtime_mode: str,
    hline_y: float = 0.8,
) -> None:
    def _first_crossing(xs: np.ndarray, mean: np.ndarray, thresh: float) -> Optional[int]:
        xs = np.asarray(xs, dtype=float)
        mean = np.asarray(mean, dtype=float)
        ok = np.isfinite(xs) & np.isfinite(mean)
        if not np.any(ok):
            return None
        xs_ok = xs[ok]
        mean_ok = mean[ok]
        hit = np.where(mean_ok >= float(thresh))[0]
        if len(hit) == 0:
            return None
        return int(xs_ok[int(hit[0])])

    def _curves(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        curves = []
        for _, dsub in df.groupby("subject_id", sort=True):
            try:
                v = _subject_cumtime_curve(dsub, max_fixations=max_fixations, mode=cumtime_mode)
            except Exception:
                continue
            if np.all(~np.isfinite(v)):
                continue
            curves.append(v)
        if len(curves) == 0:
            raise ValueError("No per-subject cumulative curves could be computed.")
        mat = np.vstack(curves)
        mean = np.nanmean(mat, axis=0)
        xs = np.arange(1, max_fixations + 1)
        return xs, mean, mat

    def _draw(ax: plt.Axes, *, df: pd.DataFrame, title: str, color: str, show_ylabel: bool) -> None:
        xs, mean, mat = _curves(df)

        # Individual curves (no error bands).
        for row in mat:
            ax.plot(xs, row, color=color, alpha=0.18, linewidth=1.0, zorder=1)

        # Group mean.
        ax.plot(xs, mean, color=color, linewidth=3.0, zorder=3)

        ax.axhline(float(hline_y), color="black", linewidth=1.5, zorder=2)

        ax.set_title(title)
        ax.set_xlabel("")
        if show_ylabel:
            ax.set_ylabel("Cumulative fixation time")
        ax.set_ylim(0.0, 1.01)
        ax.set_xlim(0.0, float(max_fixations))
        ax.set_xticks([0, 10, 20, 30])
        ax.spines[["top", "right"]].set_visible(False)

    _draw(ax_h, df=human_fix, title="Humans", color="k", show_ylabel=True)
    _draw(ax_n, df=nn_fix, title="Network", color="k", show_ylabel=False)

    # Share y-axis visually; keep ticks only on the left axis.
    ax_n.tick_params(labelleft=False)


def _save_fixation_counts_figure(
    *,
    out_dir: Path,
    tag: str,
    human_by_sub: pd.DataFrame,
    nn_by_sub: pd.DataFrame,
) -> Path:
    suffix = f"_{tag}" if tag else ""
    out_path = out_dir / f"FigureNNH_fixation_counts{suffix}.pdf"

    with plt.rc_context({"axes.labelsize": 22, "axes.titlesize": 22, "xtick.labelsize": 16, "ytick.labelsize": 16}):
        fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
        _panel_B(ax, human_by_sub, nn_by_sub)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

    return out_path


def create_figure(
    *,
    nn_root: Path,
    out_dir: Path,
    tag: str,
    max_fixations: int,
    cumtime_mode: str,
    gh_denom: str = "all",
) -> Path:
    _ensure_dir(out_dir)

    # Load revisits-by-subject tables (cached outputs).
    # Human stats live under REPO_ROOT/output/eyegaze/stats/;
    # NN stats live under nn_root/output/eyegaze/stats/.
    human_stats_dir = _REPO_ROOT / "output" / "eyegaze" / "stats"
    nn_stats_dir = nn_root / "output" / "eyegaze" / "stats"
    human_by_sub_path = _find_revisits_by_subject_csv(human_stats_dir, agent_tag="human")
    nn_by_sub_path = _find_revisits_by_subject_csv(nn_stats_dir, agent_tag="nn", tag=tag)

    human_by_sub = pd.read_csv(human_by_sub_path)
    nn_by_sub = pd.read_csv(nn_by_sub_path)

    # Load clean choice fixations for cumulative-time curve.
    human_clean_path = _find_default_clean_choice_fixations(_REPO_ROOT)
    nn_clean_path = _find_default_clean_choice_fixations(nn_root)

    human_fix = _load_clean_choice_fixations(human_clean_path)
    nn_fix = _load_clean_choice_fixations(nn_clean_path)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["xtick.major.width"] = 2
    plt.rcParams["ytick.major.width"] = 2

    with plt.rc_context({"axes.labelsize": 22, "axes.titlesize": 22, "xtick.labelsize": 16, "ytick.labelsize": 16}):
        fig = plt.figure(figsize=(24, 10))
        gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], wspace=0.55, hspace=0.60)

        axA = fig.add_subplot(gs[0, 0])

        # Panel B is split into two side-by-side axes within the same grid cell.
        gsB = gs[0, 1].subgridspec(1, 2, wspace=0.10)
        axB_h = fig.add_subplot(gsB[0, 0])
        axB_n = fig.add_subplot(gsB[0, 1], sharey=axB_h)

        axC = fig.add_subplot(gs[0, 2])
        axD = fig.add_subplot(gs[0, 3])

        axE = fig.add_subplot(gs[1, 0])
        axF = fig.add_subplot(gs[1, 1])

        # Panels G and H: each is split into top (Take) and bottom (Leave).
        gsG = gs[1, 2].subgridspec(2, 1, hspace=0.35)
        axG_take = fig.add_subplot(gsG[0, 0])
        axG_leave = fig.add_subplot(gsG[1, 0], sharex=axG_take)

        gsH = gs[1, 3].subgridspec(2, 1, hspace=0.35)
        axH_take = fig.add_subplot(gsH[0, 0])
        axH_leave = fig.add_subplot(gsH[1, 0], sharex=axH_take)

        _panel_A(axA, human_by_sub, nn_by_sub)
        _add_panel_label(axA, "A", dx=-85)

        _panel_B_split_cumtime(
            axB_h,
            axB_n,
            human_fix=human_fix,
            nn_fix=nn_fix,
            max_fixations=max_fixations,
            cumtime_mode=cumtime_mode,
            hline_y=0.8,
        )
        _add_panel_label(axB_h, "B", dx=-85)
        _add_shared_xlabel(fig, axB_h, axB_n, "Fixation number")

        # New panels C/D: duration by fixation position, relevant vs irrelevant (7+ collapsed).
        _panel_duration_by_relevance_7plus(
            axC,
            base_dir=_REPO_ROOT,
            clean_choice_fixations_path=human_clean_path,
            title="Humans",
            y_label="Fixation duration (s)",
            y_scale=1.0 / 1000.0,
            y_lim=(0.0, 1.0),
            max_fixations=8,
            keep_first_n=int(COLLAPSE_KEEP_FIRST_N),
            show_grid=False,
            colors={"relevant": "#6fc7eb", "irrelevant": "#ba7caf"},
        )
        _add_panel_label(axC, "C", dx=-85)

        _panel_duration_by_relevance_7plus(
            axD,
            base_dir=nn_root,
            clean_choice_fixations_path=nn_clean_path,
            title="Network",
            y_label="Fixation duration (steps)",
            y_lim=(1.0, 1.4),
            max_fixations=8,
            keep_first_n=int(COLLAPSE_KEEP_FIRST_N),
            show_grid=False,
            colors={"relevant": "#6fc7eb", "irrelevant": "#ba7caf"},
        )
        _add_panel_label(axD, "D", dx=-85)

        # Panels E/F: proportion relevant fixated by position (humans vs network).
        _panel_prop_relevant_fixated_by_position_7plus(
            axE,
            base_dir=_REPO_ROOT,
            clean_choice_fixations_path=human_clean_path,
            title="Humans",
            max_fixations=8,
            keep_first_n=int(COLLAPSE_KEEP_FIRST_N),
            y_label="Prop. relevant fixated",
            hline_y=0.5,
            line_color="black",
            y_lim=(0.2, 0.8),
        )
        _add_panel_label(axE, "E", dx=-85)

        _panel_prop_relevant_fixated_by_position_7plus(
            axF,
            base_dir=nn_root,
            clean_choice_fixations_path=nn_clean_path,
            title="Network",
            max_fixations=8,
            keep_first_n=int(COLLAPSE_KEEP_FIRST_N),
            y_label="Prop. relevant fixated",
            hline_y=0.5,
            line_color="black",
            y_lim=(0.2, 0.8),
        )
        _add_panel_label(axF, "F", dx=-85)

        # Panels G/H: prop positive vs negative relevant fixations, take vs leave.
        _gh_ylim = (0.1, 0.4)
        _panel_prop_positive_negative_relevant_by_decision(
            axG_take,
            axG_leave,
            base_dir=_REPO_ROOT,
            clean_choice_fixations_path=human_clean_path,
            title="Humans",
            max_fixations=int(max_fixations),
            keep_first_n=int(COLLAPSE_KEEP_FIRST_N),
            denom=gh_denom,
            y_lim=_gh_ylim,
            reward_col="reward_recalled",
        )
        _add_panel_label(axG_take, "G", dx=-85)

        # Override G panel formatting for consistent layout.
        axG_take.set_title("Take", fontsize=14)
        axG_leave.set_title("Leave", fontsize=14)
        # Re-set xticklabels (the function clears them via set_xticklabels([])
        # which affects the shared axis).
        _keep = int(COLLAPSE_KEEP_FIRST_N)
        _ticks_g = list(range(1, _keep + 2))
        axG_leave.set_xticks(_ticks_g)
        axG_leave.set_xticklabels([str(i) for i in range(1, _keep + 1)] + [f"{_keep + 1}+"])
        axG_take.tick_params(labelbottom=False)
        # Shared y-axis label centered between take and leave subplots.
        axG_take.set_ylabel("Proportion of Fixations", fontsize=22)
        axG_take.yaxis.set_label_coords(-0.18, -0.15)
        axG_leave.set_ylabel("")

        _panel_prop_positive_negative_relevant_by_decision(
            axH_take,
            axH_leave,
            base_dir=nn_root,
            clean_choice_fixations_path=nn_clean_path,
            title="Network",
            max_fixations=int(max_fixations),
            keep_first_n=int(COLLAPSE_KEEP_FIRST_N),
            denom=gh_denom,
            y_lim=_gh_ylim,
        )
        _add_panel_label(axH_take, "H", dx=-85)

        # Override H panel formatting for consistent layout.
        axH_take.set_title("Take", fontsize=14)
        axH_leave.set_title("Leave", fontsize=14)
        _ticks_h = list(range(1, _keep + 2))
        axH_leave.set_xticks(_ticks_h)
        axH_leave.set_xticklabels([str(i) for i in range(1, _keep + 1)] + [f"{_keep + 1}+"])
        axH_take.tick_params(labelbottom=False)
        # Shared y-axis label centered between take and leave subplots.
        axH_take.set_ylabel("Proportion of Fixations", fontsize=22)
        axH_take.yaxis.set_label_coords(-0.18, -0.15)
        axH_leave.set_ylabel("")

        # Overall titles for G (Humans) and H (Network) panel groups.
        axG_take.text(0.5, 1.45, "Humans", transform=axG_take.transAxes,
                      ha='center', va='bottom', fontsize=22)
        axH_take.text(0.5, 1.45, "Network", transform=axH_take.transAxes,
                      ha='center', va='bottom', fontsize=22)

        suffix = f"_{tag}" if tag else ""
        out_path = out_dir / f"FigureNNH_comparison{suffix}.pdf"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

    meta_path = out_dir / f"FigureNNH_comparison{('_' + tag) if tag else ''}.meta.txt"
    meta_path.write_text(
        "\n".join(
            [
                f"nn_root={nn_root}",
                f"out_dir={out_dir}",
                f"tag={tag}",
                f"human_stats_dir={human_stats_dir}",
                f"nn_stats_dir={nn_stats_dir}",
                f"human_by_sub={human_by_sub_path}",
                f"nn_by_sub={nn_by_sub_path}",
                f"human_clean_choice_fixations={human_clean_path}",
                f"nn_clean_choice_fixations={nn_clean_path}",
                f"max_fixations={max_fixations}",
                f"cumtime_mode={cumtime_mode}",
                f"cumtime_hline_y=0.8",
                f"duration_by_position_keep_first_n={int(COLLAPSE_KEEP_FIRST_N)}",
                "duration_by_position_max_fixations=8",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Humans vs NN comparison figure (A–H; top row implemented).")
    parser.add_argument(
        "--nn-root",
        type=str,
        default="metarnn/simulations/human_like",
        help="NN human_like root containing output/choice_fixations_clean*.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for the composite figure.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional suffix added to output filename.",
    )
    parser.add_argument(
        "--max-fixations",
        type=int,
        default=40,
        help="Max fixation number for the cumulative-time curve (default: 40).",
    )
    parser.add_argument(
        "--cumtime-mode",
        type=str,
        default="all_trials",
        choices=["all_trials", "conditional"],
        help="How to average cumulative time across trials (matches plot_fixation_count_and_cumtime.py).",
    )
    parser.add_argument(
        "--gh-denom",
        type=str,
        default="all",
        choices=["relevant", "all"],
        help="Denominator for G/H panels. 'relevant': only relevant fixations (sums to 1). "
             "'all': all fixations incl. irrelevant (like overview figure).",
    )

    args = parser.parse_args()

    out_path = create_figure(
        nn_root=Path(args.nn_root).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        tag=str(args.tag),
        max_fixations=int(args.max_fixations),
        cumtime_mode=str(args.cumtime_mode),
        gh_denom=str(args.gh_denom),
    )

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
