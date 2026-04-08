"""Analyze average fixation duration by fixation position during choice.

This analysis operates on the *clean choice fixations* CSV produced by
`analysis/prepare_choice_fixations.py` (or the NN-compiled equivalent).

For fixation positions 1..N, it computes mean fixation duration (ms):
  1) Overall
  2) Split by relevance (relevant vs irrelevant)
  3) For relevant fixations, split by absolute reward rank among the 6 items in the game
  4) For irrelevant fixations, split by absolute reward rank among the 6 items in the game

Key conventions:
- Fixation duration is taken from the `fixation_duration` column in the clean CSV.
- Absolute reward ranks (1..6) are computed within (subject, game) using |reward|,
  with ties assigned to the higher bin via rank(method='max').
- Aggregation is done per-subject first, then averaged across subjects with SEM.

Outputs are written to <base-dir>/output/ and <base-dir>/figures/.

This script is designed to run on both human data (repo root as base-dir) and
NN simulations compiled into a human-like directory layout (e.g.
`nn-simulations/human_like_02_16/`).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_EXCLUDED = ("107", "131")

# Collapsed-position plotting convention:
# - keep fixation positions 1..6
# - collapse positions 7..max_fixations into a single "7+" bin
COLLAPSE_KEEP_FIRST_N = 6


def _sem(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    n = len(x)
    if n <= 1:
        return float("nan")
    return float(x.std(ddof=1) / np.sqrt(n))


def _list_numeric_subdirs(parent: Path) -> List[str]:
    if not parent.exists():
        return []
    out: List[str] = []
    for p in sorted(parent.iterdir()):
        if p.is_dir() and p.name.isdigit():
            out.append(p.name)
    return out


def _is_image_name(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parts = value.split("_")
    return len(parts) == 4 and all(len(p) > 0 for p in parts)


def _find_default_clean_choice_fixations(base_dir: Path) -> Optional[Path]:
    out_dir = base_dir / "output"
    preferred = out_dir / "choice_fixations_clean_buffer_50.csv"
    if preferred.exists():
        return preferred
    fallback = out_dir / "choice_fixations_clean.csv"
    if fallback.exists():
        return fallback
    matches = sorted(out_dir.glob("choice_fixations_clean_buffer_*.csv"))
    return matches[0] if matches else None


def load_clean_choice_fixations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [
        "subject_id",
        "game",
        "trial_number",
        "option",
        "choice",
        "image",
        "relevance",
        "fixation_duration",
        "fixation_count",
        "reward",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Clean choice fixations missing required columns: {missing}")

    df["subject_id"] = df["subject_id"].astype(str)
    df["game"] = pd.to_numeric(df["game"], errors="coerce").astype(int)
    df["trial_number"] = pd.to_numeric(df["trial_number"], errors="coerce").astype(int)
    df["fixation_count"] = pd.to_numeric(df["fixation_count"], errors="coerce").astype(int)
    df["fixation_duration"] = pd.to_numeric(df["fixation_duration"], errors="coerce")
    df["relevance"] = pd.to_numeric(df["relevance"], errors="coerce")
    df["reward"] = pd.to_numeric(df["reward"], errors="coerce")

    df = df[df["image"].apply(_is_image_name)].copy()
    df["is_relevant"] = (pd.to_numeric(df["relevance"], errors="coerce").fillna(0.0) > 0.5).astype(int)
    df["is_irrelevant"] = (df["is_relevant"] == 0).astype(int)
    return df


def load_true_rewards(
    base_dir: Path,
    *,
    subjects: Optional[Sequence[str]] = None,
    excluded_subjects: Sequence[str] = DEFAULT_EXCLUDED,
) -> pd.DataFrame:
    """Return (subject_id, game, image) -> true encoded reward.

    Supports both:
    - human logfiles (which include encoding rows), and
    - NN compiled logfiles in human-like layout.

    We accept either encoding event 'image' or 'value' because different
    pipelines in this repo have used both.
    """

    data_dir = base_dir / "data"
    subids = _list_numeric_subdirs(data_dir)
    if subjects is not None:
        allowed = {str(s) for s in subjects}
        subids = [s for s in subids if s in allowed]

    excluded = set(map(str, excluded_subjects))
    subids = [s for s in subids if s not in excluded]

    rows: List[pd.DataFrame] = []
    for subid in subids:
        beh_file = data_dir / subid / f"{subid}_MAIN_logfile_7.csv"
        if not beh_file.exists():
            continue
        beh = pd.read_csv(beh_file)
        enc = beh[(beh.get("phase") == "encoding") & (beh.get("event").isin(["image", "value"]))].copy()
        if enc.empty or (not {"game", "image", "outcome"}.issubset(enc.columns)):
            continue
        enc = enc[["game", "image", "outcome"]].dropna(subset=["game", "image", "outcome"]).copy()
        enc["game"] = pd.to_numeric(enc["game"], errors="coerce")
        enc["outcome"] = pd.to_numeric(enc["outcome"], errors="coerce")
        enc = enc.dropna(subset=["game", "outcome"]).copy()
        if enc.empty:
            continue
        enc["game"] = enc["game"].astype(int)
        enc["image"] = enc["image"].astype(str)
        enc = enc.drop_duplicates(subset=["game", "image"], keep="first").copy()
        enc["subject_id"] = str(subid)
        enc = enc.rename(columns={"outcome": "reward_true"})
        rows.append(enc[["subject_id", "game", "image", "reward_true"]])

    if not rows:
        return pd.DataFrame(columns=["subject_id", "game", "image", "reward_true"])
    out = pd.concat(rows, ignore_index=True)
    out["subject_id"] = out["subject_id"].astype(str)
    out["game"] = pd.to_numeric(out["game"], errors="coerce").astype(int)
    out["reward_true"] = pd.to_numeric(out["reward_true"], errors="coerce")
    out = out.drop_duplicates(subset=["subject_id", "game", "image"], keep="first")
    return out


def _compute_rank_with_ties(values: pd.Series, *, n_bins: int) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce").abs().fillna(-1.0)
    r = v.rank(method="max", ascending=True).clip(lower=1, upper=n_bins)
    return r.astype(int)


def add_within_subset_absrank3(
    df: pd.DataFrame,
    true_rewards: pd.DataFrame,
) -> pd.DataFrame:
    """Add abs-rank within relevant and within irrelevant items (1..3).

    Ranks are computed within each (subject, game, option) based on |true reward|,
    separately for relevant vs irrelevant items.
    """

    if df.empty:
        return df
    if true_rewards.empty:
        raise RuntimeError("No true rewards found; cannot compute within-subset ranks.")

    opts = df[["subject_id", "game", "option"]].drop_duplicates()
    opt_items = opts.merge(true_rewards, on=["subject_id", "game"], how="left")
    opt_items = opt_items.dropna(subset=["image"]).copy()
    opt_items["image"] = opt_items["image"].astype(str)
    opt_items["is_relevant_token"] = [
        int(isinstance(opt, str) and isinstance(img, str) and opt in img.split("_"))
        for opt, img in zip(opt_items["option"].tolist(), opt_items["image"].tolist())
    ]

    rel_items = opt_items[opt_items["is_relevant_token"] == 1].copy()
    irrel_items = opt_items[opt_items["is_relevant_token"] == 0].copy()

    if not rel_items.empty:
        rel_items["rank_rel_3"] = rel_items.groupby(["subject_id", "game", "option"], sort=False)["reward_true"].transform(
            lambda s: _compute_rank_with_ties(s, n_bins=3)
        )
    if not irrel_items.empty:
        irrel_items["rank_irrel_3"] = irrel_items.groupby(["subject_id", "game", "option"], sort=False)["reward_true"].transform(
            lambda s: _compute_rank_with_ties(s, n_bins=3)
        )

    rel_ranks = rel_items[["subject_id", "game", "option", "image", "rank_rel_3"]].drop_duplicates(
        subset=["subject_id", "game", "option", "image"]
    )
    irrel_ranks = irrel_items[["subject_id", "game", "option", "image", "rank_irrel_3"]].drop_duplicates(
        subset=["subject_id", "game", "option", "image"]
    )

    out = df.merge(rel_ranks, on=["subject_id", "game", "option", "image"], how="left")
    out = out.merge(irrel_ranks, on=["subject_id", "game", "option", "image"], how="left")
    out["rank_rel_3"] = pd.to_numeric(out["rank_rel_3"], errors="coerce")
    out["rank_irrel_3"] = pd.to_numeric(out["rank_irrel_3"], errors="coerce")
    return out


def _mean_sem_by_subject(df: pd.DataFrame, *, group_cols: Sequence[str], value_col: str) -> pd.DataFrame:
    per_sub = (
        df.groupby(["subject_id", *group_cols], as_index=False)
        .agg(
            n_fixations=(value_col, "size"),
            mean_duration=(value_col, "mean"),
        )
    )
    out = per_sub.groupby(list(group_cols), as_index=False).agg(
        mean_duration=("mean_duration", "mean"),
        mean_duration_sem=("mean_duration", _sem),
        n_subjects=("subject_id", lambda s: int(pd.Series(s).nunique())),
        n_fixations=("n_fixations", "sum"),
    )
    return out


def _mean_sem_prop_by_subject(
    per_sub: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    prop_col: str,
) -> pd.DataFrame:
    """Aggregate mean + SEM across subjects for a proportion column."""

    out = per_sub.groupby(list(group_cols), as_index=False).agg(
        p_time=(prop_col, "mean"),
        p_time_sem=(prop_col, _sem),
        n_subjects=("subject_id", lambda s: int(pd.Series(s).nunique())),
        n_fixations=("n_fixations", "sum") if "n_fixations" in per_sub.columns else ("subject_id", "size"),
    )
    return out


def build_duration_summaries(
    fix: pd.DataFrame,
    *,
    max_fixations: int,
    base_dir: Path,
    excluded_subjects: Sequence[str],
) -> pd.DataFrame:
    df = fix.copy()
    df = df[df["fixation_count"] <= int(max_fixations)].copy()
    df = df.dropna(subset=["fixation_duration"]).copy()
    if df.empty:
        return pd.DataFrame()

    true_rewards = load_true_rewards(
        base_dir,
        subjects=sorted(df["subject_id"].dropna().astype(str).unique().tolist()),
        excluded_subjects=tuple(excluded_subjects),
    )
    df = add_within_subset_absrank3(df, true_rewards)
    df["fixation_position"] = pd.to_numeric(df["fixation_count"], errors="coerce").astype(int)

    frames: List[pd.DataFrame] = []

    # Panel 1: overall
    overall = _mean_sem_by_subject(df, group_cols=["fixation_position"], value_col="fixation_duration")
    overall["panel"] = "overall"
    overall["subset"] = "overall"
    overall["rank_bin"] = np.nan
    frames.append(overall)

    # Panel 2: relevant vs irrelevant
    for subset_name, mask in [
        ("relevant", df["is_relevant"] == 1),
        ("irrelevant", df["is_irrelevant"] == 1),
    ]:
        sub = df[mask].copy()
        if sub.empty:
            continue
        s = _mean_sem_by_subject(sub, group_cols=["fixation_position"], value_col="fixation_duration")
        s["panel"] = "by_relevance"
        s["subset"] = subset_name
        s["rank_bin"] = np.nan
        frames.append(s)

    # Panel 3: relevant by within-relevant absrank3
    rel = df[(df["is_relevant"] == 1) & (pd.to_numeric(df["rank_rel_3"], errors="coerce").notna())].copy()
    if not rel.empty:
        rel["rank_bin"] = pd.to_numeric(rel["rank_rel_3"], errors="coerce").astype(int)
        rel = rel[rel["rank_bin"].isin([1, 2, 3])].copy()
        r = _mean_sem_by_subject(rel, group_cols=["fixation_position", "rank_bin"], value_col="fixation_duration")
        r["panel"] = "relevant_by_within_absrank3"
        r["subset"] = "relevant"
        frames.append(r)

    # Panel 4: irrelevant by within-irrelevant absrank3
    ir = df[(df["is_irrelevant"] == 1) & (pd.to_numeric(df["rank_irrel_3"], errors="coerce").notna())].copy()
    if not ir.empty:
        ir["rank_bin"] = pd.to_numeric(ir["rank_irrel_3"], errors="coerce").astype(int)
        ir = ir[ir["rank_bin"].isin([1, 2, 3])].copy()
        i = _mean_sem_by_subject(ir, group_cols=["fixation_position", "rank_bin"], value_col="fixation_duration")
        i["panel"] = "irrelevant_by_within_absrank3"
        i["subset"] = "irrelevant"
        frames.append(i)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out["max_fixations"] = int(max_fixations)
    return out


def _collapse_positions_to_7plus(
    df: pd.DataFrame,
    *,
    trial_cols: Sequence[str],
    position_col: str,
    keep_first_n: int,
    value_cols_mean: Sequence[str],
    extra_group_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Collapse fixation positions > keep_first_n into a single tail bin (keep_first_n+1).

    Tail values are computed as within-trial means (optionally split by extra_group_cols),
    so trials with many tail fixations do not get overweighted.
    """

    if df.empty:
        return df

    d = df.copy()
    d[position_col] = pd.to_numeric(d[position_col], errors="coerce")
    d = d.dropna(subset=[position_col]).copy()
    d[position_col] = d[position_col].astype(int)

    early = d[d[position_col] <= int(keep_first_n)].copy()
    tail = d[d[position_col] > int(keep_first_n)].copy()
    if tail.empty:
        return early

    group_cols = list(trial_cols) + list(extra_group_cols)
    agg_spec = {c: (c, "mean") for c in value_cols_mean}
    tail_agg = tail.groupby(group_cols, as_index=False).agg(**agg_spec)
    tail_agg[position_col] = int(keep_first_n) + 1

    # Preserve any grouping columns used downstream (e.g., relevance flags, rank bins).
    out = pd.concat([early, tail_agg], ignore_index=True)
    out[position_col] = pd.to_numeric(out[position_col], errors="coerce").astype(int)
    return out


def build_duration_summaries_7plus(
    fix: pd.DataFrame,
    *,
    max_fixations: int,
    base_dir: Path,
    excluded_subjects: Sequence[str],
    keep_first_n: int = COLLAPSE_KEEP_FIRST_N,
) -> pd.DataFrame:
    """As in build_duration_summaries, but collapses positions 7..max_fixations into "7+".

    The 7+ point is computed as a within-trial mean over the remaining fixations.
    """

    df = fix.copy()
    df = df[df["fixation_count"] <= int(max_fixations)].copy()
    df = df.dropna(subset=["fixation_duration"]).copy()
    if df.empty:
        return pd.DataFrame()

    true_rewards = load_true_rewards(
        base_dir,
        subjects=sorted(df["subject_id"].dropna().astype(str).unique().tolist()),
        excluded_subjects=tuple(excluded_subjects),
    )
    df = add_within_subset_absrank3(df, true_rewards)
    df["fixation_position"] = pd.to_numeric(df["fixation_count"], errors="coerce").astype(int)

    trial_cols = ["subject_id", "game", "trial_number", "option"]

    frames: List[pd.DataFrame] = []
    max_plot = int(keep_first_n) + 1

    # Panel 1: overall
    df_overall = _collapse_positions_to_7plus(
        df,
        trial_cols=trial_cols,
        position_col="fixation_position",
        keep_first_n=keep_first_n,
        value_cols_mean=["fixation_duration"],
    )
    overall = _mean_sem_by_subject(df_overall, group_cols=["fixation_position"], value_col="fixation_duration")
    overall["panel"] = "overall"
    overall["subset"] = "overall"
    overall["rank_bin"] = np.nan
    frames.append(overall)

    # Panel 2: relevant vs irrelevant
    for subset_name, mask in [
        ("relevant", df["is_relevant"] == 1),
        ("irrelevant", df["is_irrelevant"] == 1),
    ]:
        sub = df[mask].copy()
        if sub.empty:
            continue
        sub = _collapse_positions_to_7plus(
            sub,
            trial_cols=trial_cols,
            position_col="fixation_position",
            keep_first_n=keep_first_n,
            value_cols_mean=["fixation_duration"],
        )
        s = _mean_sem_by_subject(sub, group_cols=["fixation_position"], value_col="fixation_duration")
        s["panel"] = "by_relevance"
        s["subset"] = subset_name
        s["rank_bin"] = np.nan
        frames.append(s)

    # Panel 3: relevant by within-relevant absrank3
    rel = df[(df["is_relevant"] == 1) & (pd.to_numeric(df["rank_rel_3"], errors="coerce").notna())].copy()
    if not rel.empty:
        rel["rank_bin"] = pd.to_numeric(rel["rank_rel_3"], errors="coerce").astype(int)
        rel = rel[rel["rank_bin"].isin([1, 2, 3])].copy()
        rel = _collapse_positions_to_7plus(
            rel,
            trial_cols=trial_cols,
            position_col="fixation_position",
            keep_first_n=keep_first_n,
            value_cols_mean=["fixation_duration"],
            extra_group_cols=["rank_bin"],
        )
        r = _mean_sem_by_subject(rel, group_cols=["fixation_position", "rank_bin"], value_col="fixation_duration")
        r["panel"] = "relevant_by_within_absrank3"
        r["subset"] = "relevant"
        frames.append(r)

    # Panel 4: irrelevant by within-irrelevant absrank3
    ir = df[(df["is_irrelevant"] == 1) & (pd.to_numeric(df["rank_irrel_3"], errors="coerce").notna())].copy()
    if not ir.empty:
        ir["rank_bin"] = pd.to_numeric(ir["rank_irrel_3"], errors="coerce").astype(int)
        ir = ir[ir["rank_bin"].isin([1, 2, 3])].copy()
        ir = _collapse_positions_to_7plus(
            ir,
            trial_cols=trial_cols,
            position_col="fixation_position",
            keep_first_n=keep_first_n,
            value_cols_mean=["fixation_duration"],
            extra_group_cols=["rank_bin"],
        )
        i = _mean_sem_by_subject(ir, group_cols=["fixation_position", "rank_bin"], value_col="fixation_duration")
        i["panel"] = "irrelevant_by_within_absrank3"
        i["subset"] = "irrelevant"
        frames.append(i)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out["max_fixations"] = int(max_plot)
    return out


def build_time_share_summaries(
    fix: pd.DataFrame,
    *,
    max_fixations: int,
    base_dir: Path,
    excluded_subjects: Sequence[str],
) -> pd.DataFrame:
    """Build proportion-of-viewing-time summaries.

    Conventions
    -----------
    - Panel 'overall': share of total early-fixation time allocated to each fixation position.
    - Panel 'by_relevance': at each position, share of time on relevant vs irrelevant (sums to 1).
    - Panel 'relevant_by_within_absrank3': within relevant time at each position, share by rank bin 1..3.
    - Panel 'irrelevant_by_within_absrank3': within irrelevant time at each position, share by rank bin 1..3.
    """

    df_all = fix.copy()
    df_all = df_all.dropna(subset=["fixation_duration"]).copy()
    if df_all.empty:
        return pd.DataFrame()

    df = df_all[df_all["fixation_count"] <= int(max_fixations)].copy()
    if df.empty:
        return pd.DataFrame()

    true_rewards = load_true_rewards(
        base_dir,
        subjects=sorted(df["subject_id"].dropna().astype(str).unique().tolist()),
        excluded_subjects=tuple(excluded_subjects),
    )
    df = add_within_subset_absrank3(df, true_rewards)
    df["fixation_position"] = pd.to_numeric(df["fixation_count"], errors="coerce").astype(int)
    subject_ids = sorted(df_all["subject_id"].dropna().astype(str).unique().tolist())
    positions = list(range(1, int(max_fixations) + 1))

    # -----------------
    # Chance baselines
    # -----------------
    # Panel 1 chance: conditional-on-reaching baseline.
    # For a trial with m early fixations (m = min(n_fix, N)), chance within that
    # trial is uniform across the m positions (1/m). The chance curve at position
    # k averages 1/m over trials that reached k.
    trial_cols = ["subject_id", "game", "trial_number", "option"]
    trial_n = (
        df.dropna(subset=["fixation_position"])  # uses early fixations only
        .groupby(trial_cols, as_index=False)
        .agg(m_fix=("fixation_position", "max"))
    )
    trial_n["m_fix"] = pd.to_numeric(trial_n["m_fix"], errors="coerce").fillna(0).astype(int)
    chance_overall_rows: List[pd.DataFrame] = []
    for k in positions:
        c = trial_n[["subject_id", "m_fix"]].copy()
        c["fixation_position"] = int(k)
        c["chance_p_time"] = np.where(
            c["m_fix"].to_numpy(dtype=int) >= int(k),
            1.0 / c["m_fix"].replace(0, np.nan).to_numpy(dtype=float),
            np.nan,
        )
        chance_overall_rows.append(c[["subject_id", "fixation_position", "chance_p_time"]])
    chance_overall = pd.concat(chance_overall_rows, ignore_index=True) if chance_overall_rows else pd.DataFrame()
    chance_overall_sub = (
        chance_overall.groupby(["subject_id", "fixation_position"], as_index=False).agg(
            chance_p_time=("chance_p_time", "mean")
        )
        if not chance_overall.empty
        else pd.DataFrame(columns=["subject_id", "fixation_position", "chance_p_time"])
    )
    chance_overall_group = (
        chance_overall_sub.groupby(["fixation_position"], as_index=False).agg(
            chance_p_time=("chance_p_time", "mean"),
            chance_p_time_sem=("chance_p_time", _sem),
        )
        if not chance_overall_sub.empty
        else pd.DataFrame(columns=["fixation_position", "chance_p_time", "chance_p_time_sem"])
    )

    # Panel 2 chance: 3 relevant / 3 irrelevant items per choice => 0.5 each.
    chance_by_relevance = {"relevant": 0.5, "irrelevant": 0.5}

    # Panel 3/4 chance: probability of rank bins under uniform sampling of an item
    # from the relevant (or irrelevant) set for that choice.
    # This accounts for ties in |reward| which can cause bins to have sizes != 1.
    opts = df[["subject_id", "game", "option"]].drop_duplicates()
    opt_items = opts.merge(true_rewards, on=["subject_id", "game"], how="left")
    opt_items = opt_items.dropna(subset=["image"]).copy()
    opt_items["image"] = opt_items["image"].astype(str)
    opt_items["is_relevant_token"] = [
        int(isinstance(opt, str) and isinstance(img, str) and opt in img.split("_"))
        for opt, img in zip(opt_items["option"].tolist(), opt_items["image"].tolist())
    ]
    rel_items = opt_items[opt_items["is_relevant_token"] == 1].copy()
    irrel_items = opt_items[opt_items["is_relevant_token"] == 0].copy()
    chance_rank_rel = {1: 1 / 3, 2: 1 / 3, 3: 1 / 3}
    chance_rank_irrel = {1: 1 / 3, 2: 1 / 3, 3: 1 / 3}
    if not rel_items.empty:
        rel_items["rank_bin"] = rel_items.groupby(["subject_id", "game", "option"], sort=False)["reward_true"].transform(
            lambda s: _compute_rank_with_ties(s, n_bins=3)
        )
        rel_choices = rel_items[["subject_id", "game", "option"]].drop_duplicates()
        rel_counts = (
            rel_items.groupby(["subject_id", "game", "option", "rank_bin"], as_index=False)
            .agg(n_items=("image", "size"))
        )
        complete = rel_choices.merge(pd.DataFrame({"rank_bin": [1, 2, 3]}), how="cross")
        rel_counts = complete.merge(rel_counts, on=["subject_id", "game", "option", "rank_bin"], how="left")
        rel_counts["n_items"] = pd.to_numeric(rel_counts["n_items"], errors="coerce").fillna(0.0)
        rel_counts["p"] = rel_counts["n_items"] / 3.0
        rel_sub = rel_counts.groupby(["subject_id", "rank_bin"], as_index=False).agg(p=("p", "mean"))
        rel_group = rel_sub.groupby(["rank_bin"], as_index=False).agg(p=("p", "mean"))
        chance_rank_rel = {int(b): float(p) for b, p in zip(rel_group["rank_bin"].tolist(), rel_group["p"].tolist())}
        for b in [1, 2, 3]:
            chance_rank_rel.setdefault(b, 0.0)
    if not irrel_items.empty:
        irrel_items["rank_bin"] = irrel_items.groupby(["subject_id", "game", "option"], sort=False)["reward_true"].transform(
            lambda s: _compute_rank_with_ties(s, n_bins=3)
        )
        ir_choices = irrel_items[["subject_id", "game", "option"]].drop_duplicates()
        ir_counts = (
            irrel_items.groupby(["subject_id", "game", "option", "rank_bin"], as_index=False)
            .agg(n_items=("image", "size"))
        )
        complete = ir_choices.merge(pd.DataFrame({"rank_bin": [1, 2, 3]}), how="cross")
        ir_counts = complete.merge(ir_counts, on=["subject_id", "game", "option", "rank_bin"], how="left")
        ir_counts["n_items"] = pd.to_numeric(ir_counts["n_items"], errors="coerce").fillna(0.0)
        ir_counts["p"] = ir_counts["n_items"] / 3.0
        ir_sub = ir_counts.groupby(["subject_id", "rank_bin"], as_index=False).agg(p=("p", "mean"))
        ir_group = ir_sub.groupby(["rank_bin"], as_index=False).agg(p=("p", "mean"))
        chance_rank_irrel = {int(b): float(p) for b, p in zip(ir_group["rank_bin"].tolist(), ir_group["p"].tolist())}
        for b in [1, 2, 3]:
            chance_rank_irrel.setdefault(b, 0.0)

    frames: List[pd.DataFrame] = []

    # Panel 1: overall share of time by fixation position, conditional on reaching.
    # For each trial, compute the within-trial proportion of early time allocated
    # to each early position (1..m), where m is that trial's # of early fixations.
    per_trial_pos = (
        df.groupby([*trial_cols, "fixation_position"], as_index=False)
        .agg(time_ms=("fixation_duration", "sum"), n_fixations=("fixation_duration", "size"))
    )
    per_trial_tot = per_trial_pos.groupby(trial_cols, as_index=False).agg(denom_time_ms=("time_ms", "sum"))
    per_trial_pos = per_trial_pos.merge(per_trial_tot, on=trial_cols, how="left")
    per_trial_pos["p_time"] = np.where(
        pd.to_numeric(per_trial_pos["denom_time_ms"], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0,
        per_trial_pos["time_ms"] / per_trial_pos["denom_time_ms"],
        np.nan,
    )
    per_sub = (
        per_trial_pos.groupby(["subject_id", "fixation_position"], as_index=False)
        .agg(p_time=("p_time", "mean"), n_fixations=("n_fixations", "sum"))
    )
    out = _mean_sem_prop_by_subject(per_sub, group_cols=["fixation_position"], prop_col="p_time")
    out = out.merge(chance_overall_group[["fixation_position", "chance_p_time"]], on="fixation_position", how="left")
    out["panel"] = "overall"
    out["subset"] = "overall"
    out["rank_bin"] = np.nan
    frames.append(out)

    # Panel 2: relevant vs irrelevant time share within each position.
    # Fill missing categories with 0 so relevant+irrelevant sums to 1 per subject-position,
    # and the *group means* also sum to 1.
    per_rel = (
        df.groupby(["subject_id", "fixation_position", "is_relevant"], as_index=False)
        .agg(time_ms=("fixation_duration", "sum"), n_fixations=("fixation_duration", "size"))
    )

    complete_rel = pd.MultiIndex.from_product(
        [subject_ids, positions, [0, 1]],
        names=["subject_id", "fixation_position", "is_relevant"],
    ).to_frame(index=False)
    per_rel = complete_rel.merge(per_rel, on=["subject_id", "fixation_position", "is_relevant"], how="left")
    per_rel["time_ms"] = pd.to_numeric(per_rel["time_ms"], errors="coerce").fillna(0.0)
    per_rel["n_fixations"] = pd.to_numeric(per_rel["n_fixations"], errors="coerce").fillna(0).astype(int)

    denom = per_rel.groupby(["subject_id", "fixation_position"], as_index=False).agg(denom_time_ms=("time_ms", "sum"))
    per_rel = per_rel.merge(denom, on=["subject_id", "fixation_position"], how="left")
    per_rel["p_time"] = np.where(
        pd.to_numeric(per_rel["denom_time_ms"], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0,
        per_rel["time_ms"] / per_rel["denom_time_ms"],
        np.nan,
    )
    per_rel["subset"] = np.where(per_rel["is_relevant"].astype(int) == 1, "relevant", "irrelevant")
    out = _mean_sem_prop_by_subject(per_rel, group_cols=["fixation_position", "subset"], prop_col="p_time")
    out["chance_p_time"] = out["subset"].map(chance_by_relevance).astype(float)
    out["panel"] = "by_relevance"
    out["rank_bin"] = np.nan
    frames.append(out)

    # Panel 3: relevant time share by within-relevant rank bin.
    rel = df[(df["is_relevant"] == 1) & (pd.to_numeric(df["rank_rel_3"], errors="coerce").notna())].copy()
    if not rel.empty:
        rel["rank_bin"] = pd.to_numeric(rel["rank_rel_3"], errors="coerce").astype(int)
        rel = rel[rel["rank_bin"].isin([1, 2, 3])].copy()
        per = (
            rel.groupby(["subject_id", "fixation_position", "rank_bin"], as_index=False)
            .agg(time_ms=("fixation_duration", "sum"), n_fixations=("fixation_duration", "size"))
        )

        complete_rank = pd.MultiIndex.from_product(
            [subject_ids, positions, [1, 2, 3]],
            names=["subject_id", "fixation_position", "rank_bin"],
        ).to_frame(index=False)
        per = complete_rank.merge(per, on=["subject_id", "fixation_position", "rank_bin"], how="left")
        per["time_ms"] = pd.to_numeric(per["time_ms"], errors="coerce").fillna(0.0)
        per["n_fixations"] = pd.to_numeric(per["n_fixations"], errors="coerce").fillna(0).astype(int)

        denom = per.groupby(["subject_id", "fixation_position"], as_index=False).agg(denom_time_ms=("time_ms", "sum"))
        per = per.merge(denom, on=["subject_id", "fixation_position"], how="left")
        per["p_time"] = np.where(
            pd.to_numeric(per["denom_time_ms"], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0,
            per["time_ms"] / per["denom_time_ms"],
            np.nan,
        )
        out = _mean_sem_prop_by_subject(per, group_cols=["fixation_position", "rank_bin"], prop_col="p_time")
        out["chance_p_time"] = out["rank_bin"].map(lambda b: float(chance_rank_rel.get(int(b), 0.0)))
        out["panel"] = "relevant_by_within_absrank3"
        out["subset"] = "relevant"
        frames.append(out)

    # Panel 4: irrelevant time share by within-irrelevant rank bin.
    ir = df[(df["is_irrelevant"] == 1) & (pd.to_numeric(df["rank_irrel_3"], errors="coerce").notna())].copy()
    if not ir.empty:
        ir["rank_bin"] = pd.to_numeric(ir["rank_irrel_3"], errors="coerce").astype(int)
        ir = ir[ir["rank_bin"].isin([1, 2, 3])].copy()
        per = (
            ir.groupby(["subject_id", "fixation_position", "rank_bin"], as_index=False)
            .agg(time_ms=("fixation_duration", "sum"), n_fixations=("fixation_duration", "size"))
        )

        complete_rank = pd.MultiIndex.from_product(
            [subject_ids, positions, [1, 2, 3]],
            names=["subject_id", "fixation_position", "rank_bin"],
        ).to_frame(index=False)
        per = complete_rank.merge(per, on=["subject_id", "fixation_position", "rank_bin"], how="left")
        per["time_ms"] = pd.to_numeric(per["time_ms"], errors="coerce").fillna(0.0)
        per["n_fixations"] = pd.to_numeric(per["n_fixations"], errors="coerce").fillna(0).astype(int)

        denom = per.groupby(["subject_id", "fixation_position"], as_index=False).agg(denom_time_ms=("time_ms", "sum"))
        per = per.merge(denom, on=["subject_id", "fixation_position"], how="left")
        per["p_time"] = np.where(
            pd.to_numeric(per["denom_time_ms"], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0,
            per["time_ms"] / per["denom_time_ms"],
            np.nan,
        )
        out = _mean_sem_prop_by_subject(per, group_cols=["fixation_position", "rank_bin"], prop_col="p_time")
        out["chance_p_time"] = out["rank_bin"].map(lambda b: float(chance_rank_irrel.get(int(b), 0.0)))
        out["panel"] = "irrelevant_by_within_absrank3"
        out["subset"] = "irrelevant"
        frames.append(out)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out["max_fixations"] = int(max_fixations)
    return out


def build_time_share_summaries_7plus(
    fix: pd.DataFrame,
    *,
    max_fixations: int,
    base_dir: Path,
    excluded_subjects: Sequence[str],
    keep_first_n: int = COLLAPSE_KEEP_FIRST_N,
) -> pd.DataFrame:
    """As in build_time_share_summaries, but collapses positions 7..max_fixations into "7+".

    The 7+ point is computed within each trial by averaging (or time-weighted averaging,
    where appropriate) over the remaining fixations.
    """

    df_all = fix.copy()
    df_all = df_all.dropna(subset=["fixation_duration"]).copy()
    if df_all.empty:
        return pd.DataFrame()

    df = df_all[df_all["fixation_count"] <= int(max_fixations)].copy()
    if df.empty:
        return pd.DataFrame()

    true_rewards = load_true_rewards(
        base_dir,
        subjects=sorted(df["subject_id"].dropna().astype(str).unique().tolist()),
        excluded_subjects=tuple(excluded_subjects),
    )
    df = add_within_subset_absrank3(df, true_rewards)
    df["fixation_position"] = pd.to_numeric(df["fixation_count"], errors="coerce").astype(int)

    trial_cols = ["subject_id", "game", "trial_number", "option"]
    max_plot = int(keep_first_n) + 1
    positions_plot = list(range(1, max_plot + 1))

    # -----------------
    # Chance baselines
    # -----------------
    trial_n = (
        df.dropna(subset=["fixation_position"]).groupby(trial_cols, as_index=False).agg(m_fix=("fixation_position", "max"))
    )
    trial_n["m_fix"] = pd.to_numeric(trial_n["m_fix"], errors="coerce").fillna(0).astype(int)

    chance_rows: List[pd.DataFrame] = []
    for k in positions_plot:
        c = trial_n[["subject_id", "m_fix"]].copy()
        c["fixation_position"] = int(k)
        if int(k) <= int(keep_first_n):
            reached = c["m_fix"].to_numpy(dtype=int) >= int(k)
        else:
            reached = c["m_fix"].to_numpy(dtype=int) >= int(keep_first_n) + 1
        c["chance_p_time"] = np.where(
            reached,
            1.0 / c["m_fix"].replace(0, np.nan).to_numpy(dtype=float),
            np.nan,
        )
        chance_rows.append(c[["subject_id", "fixation_position", "chance_p_time"]])

    chance_overall = pd.concat(chance_rows, ignore_index=True) if chance_rows else pd.DataFrame()
    chance_overall_sub = (
        chance_overall.groupby(["subject_id", "fixation_position"], as_index=False).agg(chance_p_time=("chance_p_time", "mean"))
        if not chance_overall.empty
        else pd.DataFrame(columns=["subject_id", "fixation_position", "chance_p_time"])
    )
    chance_overall_group = (
        chance_overall_sub.groupby(["fixation_position"], as_index=False)
        .agg(chance_p_time=("chance_p_time", "mean"), chance_p_time_sem=("chance_p_time", _sem))
        if not chance_overall_sub.empty
        else pd.DataFrame(columns=["fixation_position", "chance_p_time", "chance_p_time_sem"])
    )

    chance_by_relevance = {"relevant": 0.5, "irrelevant": 0.5}

    # Rank-bin chance baselines (same as the original function)
    opts = df[["subject_id", "game", "option"]].drop_duplicates()
    opt_items = opts.merge(true_rewards, on=["subject_id", "game"], how="left")
    opt_items = opt_items.dropna(subset=["image"]).copy()
    opt_items["image"] = opt_items["image"].astype(str)
    opt_items["is_relevant_token"] = [
        int(isinstance(opt, str) and isinstance(img, str) and opt in img.split("_"))
        for opt, img in zip(opt_items["option"].tolist(), opt_items["image"].tolist())
    ]
    rel_items = opt_items[opt_items["is_relevant_token"] == 1].copy()
    irrel_items = opt_items[opt_items["is_relevant_token"] == 0].copy()
    chance_rank_rel = {1: 1 / 3, 2: 1 / 3, 3: 1 / 3}
    chance_rank_irrel = {1: 1 / 3, 2: 1 / 3, 3: 1 / 3}
    if not rel_items.empty:
        rel_items["rank_bin"] = rel_items.groupby(["subject_id", "game", "option"], sort=False)["reward_true"].transform(
            lambda s: _compute_rank_with_ties(s, n_bins=3)
        )
        rel_choices = rel_items[["subject_id", "game", "option"]].drop_duplicates()
        rel_counts = rel_items.groupby(["subject_id", "game", "option", "rank_bin"], as_index=False).agg(n_items=("image", "size"))
        complete = rel_choices.merge(pd.DataFrame({"rank_bin": [1, 2, 3]}), how="cross")
        rel_counts = complete.merge(rel_counts, on=["subject_id", "game", "option", "rank_bin"], how="left")
        rel_counts["n_items"] = pd.to_numeric(rel_counts["n_items"], errors="coerce").fillna(0.0)
        rel_counts["p"] = rel_counts["n_items"] / 3.0
        rel_sub = rel_counts.groupby(["subject_id", "rank_bin"], as_index=False).agg(p=("p", "mean"))
        rel_group = rel_sub.groupby(["rank_bin"], as_index=False).agg(p=("p", "mean"))
        chance_rank_rel = {int(b): float(p) for b, p in zip(rel_group["rank_bin"].tolist(), rel_group["p"].tolist())}
        for b in [1, 2, 3]:
            chance_rank_rel.setdefault(b, 0.0)
    if not irrel_items.empty:
        irrel_items["rank_bin"] = irrel_items.groupby(["subject_id", "game", "option"], sort=False)["reward_true"].transform(
            lambda s: _compute_rank_with_ties(s, n_bins=3)
        )
        ir_choices = irrel_items[["subject_id", "game", "option"]].drop_duplicates()
        ir_counts = irrel_items.groupby(["subject_id", "game", "option", "rank_bin"], as_index=False).agg(n_items=("image", "size"))
        complete = ir_choices.merge(pd.DataFrame({"rank_bin": [1, 2, 3]}), how="cross")
        ir_counts = complete.merge(ir_counts, on=["subject_id", "game", "option", "rank_bin"], how="left")
        ir_counts["n_items"] = pd.to_numeric(ir_counts["n_items"], errors="coerce").fillna(0.0)
        ir_counts["p"] = ir_counts["n_items"] / 3.0
        ir_sub = ir_counts.groupby(["subject_id", "rank_bin"], as_index=False).agg(p=("p", "mean"))
        ir_group = ir_sub.groupby(["rank_bin"], as_index=False).agg(p=("p", "mean"))
        chance_rank_irrel = {int(b): float(p) for b, p in zip(ir_group["rank_bin"].tolist(), ir_group["p"].tolist())}
        for b in [1, 2, 3]:
            chance_rank_irrel.setdefault(b, 0.0)

    frames: List[pd.DataFrame] = []

    # -----------------
    # Panel 1: overall time share by (collapsed) fixation position
    # -----------------
    per_trial_pos = (
        df.groupby([*trial_cols, "fixation_position"], as_index=False)
        .agg(time_ms=("fixation_duration", "sum"), n_fixations=("fixation_duration", "size"))
    )
    per_trial_tot = per_trial_pos.groupby(trial_cols, as_index=False).agg(denom_time_ms=("time_ms", "sum"))
    per_trial_pos = per_trial_pos.merge(per_trial_tot, on=trial_cols, how="left")
    per_trial_pos["p_time"] = np.where(
        pd.to_numeric(per_trial_pos["denom_time_ms"], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0,
        per_trial_pos["time_ms"] / per_trial_pos["denom_time_ms"],
        np.nan,
    )

    per_trial_pos = per_trial_pos[pd.to_numeric(per_trial_pos["fixation_position"], errors="coerce").notna()].copy()
    per_trial_pos["fixation_position"] = pd.to_numeric(per_trial_pos["fixation_position"], errors="coerce").astype(int)

    early = per_trial_pos[per_trial_pos["fixation_position"] <= int(keep_first_n)].copy()
    tail = per_trial_pos[per_trial_pos["fixation_position"] > int(keep_first_n)].copy()
    if not tail.empty:
        tail = tail.groupby(trial_cols, as_index=False).agg(p_time=("p_time", "mean"), n_fixations=("n_fixations", "sum"))
        tail["fixation_position"] = int(keep_first_n) + 1
        per_trial_pos_collapsed = pd.concat([early[[*trial_cols, "fixation_position", "p_time", "n_fixations"]], tail], ignore_index=True)
    else:
        per_trial_pos_collapsed = early[[*trial_cols, "fixation_position", "p_time", "n_fixations"]].copy()

    per_sub = (
        per_trial_pos_collapsed.groupby(["subject_id", "fixation_position"], as_index=False)
        .agg(p_time=("p_time", "mean"), n_fixations=("n_fixations", "sum"))
    )
    out = _mean_sem_prop_by_subject(per_sub, group_cols=["fixation_position"], prop_col="p_time")
    out = out.merge(chance_overall_group[["fixation_position", "chance_p_time"]], on="fixation_position", how="left")
    out["panel"] = "overall"
    out["subset"] = "overall"
    out["rank_bin"] = np.nan
    frames.append(out)

    # -----------------
    # Panel 2: relevant vs irrelevant time share within (collapsed) position
    # -----------------
    rel_ind = df[[*trial_cols, "fixation_position", "is_relevant", "fixation_duration"]].copy()
    rel_ind["fixation_position"] = pd.to_numeric(rel_ind["fixation_position"], errors="coerce").astype(int)
    rel_ind["is_relevant"] = pd.to_numeric(rel_ind["is_relevant"], errors="coerce").fillna(0).astype(int)
    rel_ind["fixation_duration"] = pd.to_numeric(rel_ind["fixation_duration"], errors="coerce")

    # collapse to 7+: time-weighted within trial
    early = rel_ind[rel_ind["fixation_position"] <= int(keep_first_n)].copy()
    tail = rel_ind[rel_ind["fixation_position"] > int(keep_first_n)].copy()
    if not tail.empty:
        tail2 = tail.copy()
        tail2["rel_time"] = np.where(tail2["is_relevant"].astype(int) == 1, tail2["fixation_duration"], 0.0)
        tail_agg = tail2.groupby(trial_cols, as_index=False).agg(
            rel_time=("rel_time", "sum"),
            tot_time=("fixation_duration", "sum"),
        )
        tail_agg["p_relevant"] = np.where(
            pd.to_numeric(tail_agg["tot_time"], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0,
            tail_agg["rel_time"] / tail_agg["tot_time"],
            np.nan,
        )
        tail_agg["fixation_position"] = int(keep_first_n) + 1
        tail_trial = tail_agg[[*trial_cols, "fixation_position", "p_relevant"]].copy()
    else:
        tail_trial = pd.DataFrame(columns=[*trial_cols, "fixation_position", "p_relevant"])

    # For early positions: p_relevant is 1 if the fixation is relevant else 0 (time share at that position).
    early_trial = early[[*trial_cols, "fixation_position", "is_relevant"]].copy()
    early_trial["p_relevant"] = early_trial["is_relevant"].astype(float)
    early_trial = early_trial[[*trial_cols, "fixation_position", "p_relevant"]]

    per_trial_rel = pd.concat([early_trial, tail_trial], ignore_index=True)
    if not per_trial_rel.empty:
        per_sub_rel = per_trial_rel.groupby(["subject_id", "fixation_position"], as_index=False).agg(p_relevant=("p_relevant", "mean"))
        # Expand to both subsets so they sum to 1.
        per_sub_rel["subset"] = "relevant"
        per_sub_ir = per_sub_rel.copy()
        per_sub_ir["subset"] = "irrelevant"
        per_sub_ir["p_time"] = 1.0 - pd.to_numeric(per_sub_rel["p_relevant"], errors="coerce")
        per_sub_rel["p_time"] = pd.to_numeric(per_sub_rel["p_relevant"], errors="coerce")
        per_sub = pd.concat([per_sub_rel[["subject_id", "fixation_position", "subset", "p_time"]], per_sub_ir[["subject_id", "fixation_position", "subset", "p_time"]]], ignore_index=True)
    else:
        per_sub = pd.DataFrame(columns=["subject_id", "fixation_position", "subset", "p_time"])

    out = _mean_sem_prop_by_subject(per_sub, group_cols=["fixation_position", "subset"], prop_col="p_time")
    out["chance_p_time"] = out["subset"].map(chance_by_relevance).astype(float)
    out["panel"] = "by_relevance"
    out["rank_bin"] = np.nan
    frames.append(out)

    # -----------------
    # Panel 3/4: within-subset rank-bin time share within (collapsed) position
    # -----------------
    def _within_subset_rank_panel(which: str) -> Optional[pd.DataFrame]:
        if which == "relevant":
            sub = df[(df["is_relevant"] == 1) & (pd.to_numeric(df["rank_rel_3"], errors="coerce").notna())].copy()
            rank_col = "rank_rel_3"
            chance_rank = chance_rank_rel
            panel = "relevant_by_within_absrank3"
        else:
            sub = df[(df["is_irrelevant"] == 1) & (pd.to_numeric(df["rank_irrel_3"], errors="coerce").notna())].copy()
            rank_col = "rank_irrel_3"
            chance_rank = chance_rank_irrel
            panel = "irrelevant_by_within_absrank3"

        if sub.empty:
            return None

        sub["rank_bin"] = pd.to_numeric(sub[rank_col], errors="coerce").astype(int)
        sub = sub[sub["rank_bin"].isin([1, 2, 3])].copy()
        if sub.empty:
            return None

        sub["fixation_position"] = pd.to_numeric(sub["fixation_position"], errors="coerce").astype(int)
        sub["fixation_duration"] = pd.to_numeric(sub["fixation_duration"], errors="coerce")

        # Per trial × position × bin time, then within-trial share.
        per = (
            sub.groupby([*trial_cols, "fixation_position", "rank_bin"], as_index=False)
            .agg(time_ms=("fixation_duration", "sum"), n_fixations=("fixation_duration", "size"))
        )
        denom = per.groupby([*trial_cols, "fixation_position"], as_index=False).agg(denom_time_ms=("time_ms", "sum"))
        per = per.merge(denom, on=[*trial_cols, "fixation_position"], how="left")
        per["p_time"] = np.where(
            pd.to_numeric(per["denom_time_ms"], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0,
            per["time_ms"] / per["denom_time_ms"],
            np.nan,
        )

        # Collapse to 7+ by averaging p_time within trial across tail positions.
        early = per[per["fixation_position"] <= int(keep_first_n)].copy()
        tail = per[per["fixation_position"] > int(keep_first_n)].copy()
        if not tail.empty:
            tail = tail.groupby([*trial_cols, "rank_bin"], as_index=False).agg(p_time=("p_time", "mean"), n_fixations=("n_fixations", "sum"))
            tail["fixation_position"] = int(keep_first_n) + 1
            per_c = pd.concat([early[[*trial_cols, "fixation_position", "rank_bin", "p_time", "n_fixations"]], tail], ignore_index=True)
        else:
            per_c = early[[*trial_cols, "fixation_position", "rank_bin", "p_time", "n_fixations"]].copy()

        # Fill missing bins with 0 wherever a trial contributes at that (collapsed) position.
        trial_pos = per_c[[*trial_cols, "fixation_position"]].drop_duplicates()
        complete = trial_pos.merge(pd.DataFrame({"rank_bin": [1, 2, 3]}), how="cross")
        per_c = complete.merge(per_c, on=[*trial_cols, "fixation_position", "rank_bin"], how="left")
        per_c["p_time"] = pd.to_numeric(per_c["p_time"], errors="coerce").fillna(0.0)

        per_sub = per_c.groupby(["subject_id", "fixation_position", "rank_bin"], as_index=False).agg(
            p_time=("p_time", "mean"),
            n_fixations=("p_time", "size"),
        )
        out = _mean_sem_prop_by_subject(per_sub, group_cols=["fixation_position", "rank_bin"], prop_col="p_time")
        out["chance_p_time"] = out["rank_bin"].map(lambda b: float(chance_rank.get(int(b), 0.0)))
        out["panel"] = panel
        out["subset"] = which
        return out

    r = _within_subset_rank_panel("relevant")
    if r is not None and not r.empty:
        frames.append(r)
    i = _within_subset_rank_panel("irrelevant")
    if i is not None and not i.empty:
        frames.append(i)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out["max_fixations"] = int(max_plot)
    return out


def plot_time_share_summaries(
    df: pd.DataFrame,
    *,
    max_fixations: int,
    out_png: Path,
    out_pdf: Path,
    xtick_labels: Optional[Sequence[str]] = None,
) -> None:
    if df is None or df.empty:
        return

    def _set_prob_ylim(ax: plt.Axes, y_values: Iterable[float]) -> None:
        vals = [float(v) for v in y_values if v is not None and np.isfinite(v)]
        if not vals:
            ax.set_ylim(0.0, 1.0)
            return
        ymax = max(vals)
        upper = min(1.0, max(0.05, ymax * 1.10, ymax + 0.02))
        ax.set_ylim(0.0, upper)

    bin_colors = {1: "tab:gray", 2: "tab:blue", 3: "tab:red"}

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), constrained_layout=True)

    # Panel 1: overall share of time by position
    ax = axes[0]
    s = df[df["panel"] == "overall"].copy().sort_values("fixation_position")
    if not s.empty:
        x = s["fixation_position"].to_numpy(dtype=int)
        y = pd.to_numeric(s["p_time"], errors="coerce").to_numpy(dtype=float)
        se = pd.to_numeric(s.get("p_time_sem"), errors="coerce").to_numpy(dtype=float)
        ax.plot(x, y, marker="o", color="black")
        if "chance_p_time" in s.columns and pd.to_numeric(s["chance_p_time"], errors="coerce").notna().any():
            y_ch = pd.to_numeric(s["chance_p_time"], errors="coerce").to_numpy(dtype=float)
            ax.plot(x, y_ch, linestyle="--", color="black", linewidth=1.0, alpha=0.75, label="chance")
        if np.isfinite(se).any():
            ax.fill_between(x, np.clip(y - se, 0.0, 1.0), np.clip(y + se, 0.0, 1.0), color="black", alpha=0.15)
        _set_prob_ylim(ax, np.concatenate([y, np.clip(y + se, 0.0, 1.0)]))
    ax.set_title("Overall")
    ax.set_xlabel("Fixation position")
    ax.set_ylabel("Proportion of viewing time")
    ticks = list(range(1, int(max_fixations) + 1))
    ax.set_xticks(ticks)
    if xtick_labels is not None and len(xtick_labels) == len(ticks):
        ax.set_xticklabels(list(xtick_labels))
    ax.grid(True, alpha=0.3)
    if not s.empty and "chance_p_time" in s.columns and pd.to_numeric(s["chance_p_time"], errors="coerce").notna().any():
        ax.legend(frameon=False, fontsize=9)

    # Panel 2: relevant vs irrelevant within position
    ax = axes[1]
    y_for_ylim: List[float] = []
    for subset_name, color in [("relevant", "tab:blue"), ("irrelevant", "tab:orange")]:
        s = df[(df["panel"] == "by_relevance") & (df["subset"] == subset_name)].copy().sort_values("fixation_position")
        if s.empty:
            continue
        x = s["fixation_position"].to_numpy(dtype=int)
        y = pd.to_numeric(s["p_time"], errors="coerce").to_numpy(dtype=float)
        se = pd.to_numeric(s.get("p_time_sem"), errors="coerce").to_numpy(dtype=float)
        ax.plot(x, y, marker="o", color=color, label=subset_name)
        if "chance_p_time" in s.columns and pd.to_numeric(s["chance_p_time"], errors="coerce").notna().any():
            ch = float(pd.to_numeric(s["chance_p_time"], errors="coerce").dropna().iloc[0])
            ax.axhline(ch, linestyle=":", color=color, linewidth=1.0, alpha=0.6)
        if np.isfinite(se).any():
            ax.fill_between(x, np.clip(y - se, 0.0, 1.0), np.clip(y + se, 0.0, 1.0), color=color, alpha=0.15)
            y_for_ylim.extend([float(v) for v in np.clip(y + se, 0.0, 1.0) if np.isfinite(v)])
        y_for_ylim.extend([float(v) for v in y if np.isfinite(v)])
    _set_prob_ylim(ax, y_for_ylim)
    ax.set_title("Relevant vs irrelevant")
    ax.set_xlabel("Fixation position")
    ax.set_ylabel("Proportion of viewing time")
    ticks = list(range(1, int(max_fixations) + 1))
    ax.set_xticks(ticks)
    if xtick_labels is not None and len(xtick_labels) == len(ticks):
        ax.set_xticklabels(list(xtick_labels))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    # Panel 3: relevant by within-rank
    ax = axes[2]
    s0 = df[df["panel"] == "relevant_by_within_absrank3"].copy()
    y_for_ylim = []
    for b in [1, 2, 3]:
        s = s0[pd.to_numeric(s0.get("rank_bin"), errors="coerce") == b].copy().sort_values("fixation_position")
        if s.empty:
            continue
        x = s["fixation_position"].to_numpy(dtype=int)
        y = pd.to_numeric(s["p_time"], errors="coerce").to_numpy(dtype=float)
        se = pd.to_numeric(s.get("p_time_sem"), errors="coerce").to_numpy(dtype=float)
        col = bin_colors.get(b, "tab:gray")
        ax.plot(x, y, marker="o", linewidth=1.2, color=col, label=f"bin {b}")
        if "chance_p_time" in s.columns and pd.to_numeric(s["chance_p_time"], errors="coerce").notna().any():
            ch = float(pd.to_numeric(s["chance_p_time"], errors="coerce").dropna().iloc[0])
            ax.axhline(ch, linestyle=":", color=col, linewidth=1.0, alpha=0.6)
        if np.isfinite(se).any():
            ax.fill_between(x, np.clip(y - se, 0.0, 1.0), np.clip(y + se, 0.0, 1.0), color=col, alpha=0.12)
            y_for_ylim.extend([float(v) for v in np.clip(y + se, 0.0, 1.0) if np.isfinite(v)])
        y_for_ylim.extend([float(v) for v in y if np.isfinite(v)])
    _set_prob_ylim(ax, y_for_ylim)
    ax.set_title("Relevant: by within-relevant |reward| rank (1–3)")
    ax.set_xlabel("Fixation position")
    ax.set_ylabel("Proportion of viewing time")
    ticks = list(range(1, int(max_fixations) + 1))
    ax.set_xticks(ticks)
    if xtick_labels is not None and len(xtick_labels) == len(ticks):
        ax.set_xticklabels(list(xtick_labels))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8, ncol=1)

    # Panel 4: irrelevant by within-rank
    ax = axes[3]
    s0 = df[df["panel"] == "irrelevant_by_within_absrank3"].copy()
    y_for_ylim = []
    for b in [1, 2, 3]:
        s = s0[pd.to_numeric(s0.get("rank_bin"), errors="coerce") == b].copy().sort_values("fixation_position")
        if s.empty:
            continue
        x = s["fixation_position"].to_numpy(dtype=int)
        y = pd.to_numeric(s["p_time"], errors="coerce").to_numpy(dtype=float)
        se = pd.to_numeric(s.get("p_time_sem"), errors="coerce").to_numpy(dtype=float)
        col = bin_colors.get(b, "tab:gray")
        ax.plot(x, y, marker="o", linewidth=1.2, color=col, label=f"bin {b}")
        if "chance_p_time" in s.columns and pd.to_numeric(s["chance_p_time"], errors="coerce").notna().any():
            ch = float(pd.to_numeric(s["chance_p_time"], errors="coerce").dropna().iloc[0])
            ax.axhline(ch, linestyle=":", color=col, linewidth=1.0, alpha=0.6)
        if np.isfinite(se).any():
            ax.fill_between(x, np.clip(y - se, 0.0, 1.0), np.clip(y + se, 0.0, 1.0), color=col, alpha=0.12)
            y_for_ylim.extend([float(v) for v in np.clip(y + se, 0.0, 1.0) if np.isfinite(v)])
        y_for_ylim.extend([float(v) for v in y if np.isfinite(v)])
    _set_prob_ylim(ax, y_for_ylim)
    ax.set_title("Irrelevant: by within-irrelevant |reward| rank (1–3)")
    ax.set_xlabel("Fixation position")
    ax.set_ylabel("Proportion of viewing time")
    ticks = list(range(1, int(max_fixations) + 1))
    ax.set_xticks(ticks)
    if xtick_labels is not None and len(xtick_labels) == len(ticks):
        ax.set_xticklabels(list(xtick_labels))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8, ncol=1)

    fig.suptitle("Fixation viewing-time share by fixation position")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_duration_summaries(
    df: pd.DataFrame,
    *,
    max_fixations: int,
    out_png: Path,
    out_pdf: Path,
    xtick_labels: Optional[Sequence[str]] = None,
) -> None:
    if df is None or df.empty:
        return

    def _set_ylim(ax: plt.Axes, y_values: Iterable[float]) -> None:
        vals = [float(v) for v in y_values if v is not None and np.isfinite(v)]
        if not vals:
            return
        ymin = max(0.0, min(vals) * 0.95)
        ymax = max(vals) * 1.10
        ax.set_ylim(ymin, ymax)

    bin_colors = {1: "tab:gray", 2: "tab:blue", 3: "tab:red"}

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), constrained_layout=True)

    # Panel 1: overall
    ax = axes[0]
    s = df[df["panel"] == "overall"].copy().sort_values("fixation_position")
    if not s.empty:
        x = s["fixation_position"].to_numpy(dtype=int)
        y = s["mean_duration"].to_numpy(dtype=float)
        se = pd.to_numeric(s.get("mean_duration_sem"), errors="coerce").to_numpy(dtype=float)
        ax.plot(x, y, marker="o", color="black", label="overall")
        if np.isfinite(se).any():
            ax.fill_between(x, y - se, y + se, color="black", alpha=0.15)
        _set_ylim(ax, np.concatenate([y, y + se]))
    ax.set_title("Overall")
    ax.set_xlabel("Fixation position")
    ax.set_ylabel("Mean fixation duration (ms)")
    ticks = list(range(1, int(max_fixations) + 1))
    ax.set_xticks(ticks)
    if xtick_labels is not None and len(xtick_labels) == len(ticks):
        ax.set_xticklabels(list(xtick_labels))
    ax.grid(True, alpha=0.3)

    # Panel 2: relevant vs irrelevant
    ax = axes[1]
    for subset_name, color in [("relevant", "tab:blue"), ("irrelevant", "tab:orange")]:
        s = df[(df["panel"] == "by_relevance") & (df["subset"] == subset_name)].copy().sort_values("fixation_position")
        if s.empty:
            continue
        x = s["fixation_position"].to_numpy(dtype=int)
        y = s["mean_duration"].to_numpy(dtype=float)
        se = pd.to_numeric(s.get("mean_duration_sem"), errors="coerce").to_numpy(dtype=float)
        ax.plot(x, y, marker="o", color=color, label=subset_name)
        if np.isfinite(se).any():
            ax.fill_between(x, y - se, y + se, color=color, alpha=0.15)
    ax.set_title("Relevant vs irrelevant")
    ax.set_xlabel("Fixation position")
    ax.set_ylabel("Mean fixation duration (ms)")
    ticks = list(range(1, int(max_fixations) + 1))
    ax.set_xticks(ticks)
    if xtick_labels is not None and len(xtick_labels) == len(ticks):
        ax.set_xticklabels(list(xtick_labels))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    # Panel 3: relevant by within-relevant absrank3
    ax = axes[2]
    s0 = df[df["panel"] == "relevant_by_within_absrank3"].copy()
    y_for_ylim: List[float] = []
    for b in [1, 2, 3]:
        s = s0[pd.to_numeric(s0.get("rank_bin"), errors="coerce") == b].copy().sort_values("fixation_position")
        if s.empty:
            continue
        x = s["fixation_position"].to_numpy(dtype=int)
        y = s["mean_duration"].to_numpy(dtype=float)
        se = pd.to_numeric(s.get("mean_duration_sem"), errors="coerce").to_numpy(dtype=float)
        col = bin_colors.get(b, "tab:gray")
        ax.plot(x, y, marker="o", linewidth=1.2, color=col, label=f"bin {b}")
        if np.isfinite(se).any():
            ax.fill_between(x, y - se, y + se, color=col, alpha=0.12)
            y_for_ylim.extend([float(v) for v in (y + se) if np.isfinite(v)])
        y_for_ylim.extend([float(v) for v in y if np.isfinite(v)])
    if y_for_ylim:
        _set_ylim(ax, y_for_ylim)
    ax.set_title("Relevant: by within-relevant |reward| rank (1–3)")
    ax.set_xlabel("Fixation position")
    ax.set_ylabel("Mean fixation duration (ms)")
    ticks = list(range(1, int(max_fixations) + 1))
    ax.set_xticks(ticks)
    if xtick_labels is not None and len(xtick_labels) == len(ticks):
        ax.set_xticklabels(list(xtick_labels))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8, ncol=1)

    # Panel 4: irrelevant by within-irrelevant absrank3
    ax = axes[3]
    s0 = df[df["panel"] == "irrelevant_by_within_absrank3"].copy()
    y_for_ylim = []
    for b in [1, 2, 3]:
        s = s0[pd.to_numeric(s0.get("rank_bin"), errors="coerce") == b].copy().sort_values("fixation_position")
        if s.empty:
            continue
        x = s["fixation_position"].to_numpy(dtype=int)
        y = s["mean_duration"].to_numpy(dtype=float)
        se = pd.to_numeric(s.get("mean_duration_sem"), errors="coerce").to_numpy(dtype=float)
        col = bin_colors.get(b, "tab:gray")
        ax.plot(x, y, marker="o", linewidth=1.2, color=col, label=f"bin {b}")
        if np.isfinite(se).any():
            ax.fill_between(x, y - se, y + se, color=col, alpha=0.12)
            y_for_ylim.extend([float(v) for v in (y + se) if np.isfinite(v)])
        y_for_ylim.extend([float(v) for v in y if np.isfinite(v)])
    if y_for_ylim:
        _set_ylim(ax, y_for_ylim)
    ax.set_title("Irrelevant: by within-irrelevant |reward| rank (1–3)")
    ax.set_xlabel("Fixation position")
    ax.set_ylabel("Mean fixation duration (ms)")
    ticks = list(range(1, int(max_fixations) + 1))
    ax.set_xticks(ticks)
    if xtick_labels is not None and len(xtick_labels) == len(ticks):
        ax.set_xticklabels(list(xtick_labels))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8, ncol=1)

    fig.suptitle("Fixation duration by fixation position")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze mean fixation duration by fixation position (overall, relevance split, and abs-rank split). "
            "Works for both humans and NN compiled outputs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2]),
        help="Project base directory (default: repo root)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Clean choice-fixations CSV (default: auto-detect under <base-dir>/output/)",
    )
    parser.add_argument(
        "--max-fixations",
        type=int,
        default=8,
        help="Number of early fixations per trial to include.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subset of subject IDs.",
    )
    parser.add_argument(
        "--exclude-subjects",
        nargs="*",
        default=list(DEFAULT_EXCLUDED),
        help="Subjects to exclude (default drops 107 and 131 for eye analyses).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag appended to output filenames.",
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    in_path = Path(args.input) if args.input else _find_default_clean_choice_fixations(base_dir)
    if in_path is None or (not in_path.exists()):
        raise FileNotFoundError(
            "Could not find a clean choice-fixations CSV. "
            "Run `python analysis/prepare_choice_fixations.py` first, or pass --input explicitly."
        )

    fix = load_clean_choice_fixations(in_path)

    excluded = set(map(str, args.exclude_subjects))
    if excluded:
        fix = fix[~fix["subject_id"].isin(excluded)].copy()

    if args.subjects is not None and len(args.subjects) > 0:
        allowed = {str(s) for s in args.subjects}
        fix = fix[fix["subject_id"].isin(allowed)].copy()

    if fix.empty:
        raise RuntimeError("No fixations left after filtering subjects/exclusions.")

    out_dir = base_dir / "output"
    fig_dir = base_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tag = str(args.tag).strip()
    tag_sfx = f"_{tag}" if tag else ""

    summary = build_duration_summaries(
        fix,
        max_fixations=int(args.max_fixations),
        base_dir=base_dir,
        excluded_subjects=tuple(args.exclude_subjects),
    )
    if summary is None or summary.empty:
        raise RuntimeError("No duration summary produced (empty after filtering?).")

    out_csv = out_dir / f"fixation_duration_by_position_long_max{int(args.max_fixations)}{tag_sfx}.csv"
    out_png = fig_dir / f"fixation_duration_by_position_max{int(args.max_fixations)}{tag_sfx}.png"
    out_pdf = fig_dir / f"fixation_duration_by_position_max{int(args.max_fixations)}{tag_sfx}.pdf"

    summary.to_csv(out_csv, index=False)
    plot_duration_summaries(summary, max_fixations=int(args.max_fixations), out_png=out_png, out_pdf=out_pdf)

    # Additional figure: collapse positions to 1..6 and 7+ (trial-tail-averaged)
    xticks_7plus = [str(i) for i in range(1, COLLAPSE_KEEP_FIRST_N + 1)] + ["7+"]
    summary_7p = build_duration_summaries_7plus(
        fix,
        max_fixations=int(args.max_fixations),
        base_dir=base_dir,
        excluded_subjects=tuple(args.exclude_subjects),
        keep_first_n=COLLAPSE_KEEP_FIRST_N,
    )
    if summary_7p is not None and (not summary_7p.empty):
        out_7p_png = fig_dir / f"fixation_duration_by_position_7plus_max{int(args.max_fixations)}{tag_sfx}.png"
        out_7p_pdf = fig_dir / f"fixation_duration_by_position_7plus_max{int(args.max_fixations)}{tag_sfx}.pdf"
        plot_duration_summaries(
            summary_7p,
            max_fixations=int(COLLAPSE_KEEP_FIRST_N) + 1,
            out_png=out_7p_png,
            out_pdf=out_7p_pdf,
            xtick_labels=xticks_7plus,
        )

    prop = build_time_share_summaries(
        fix,
        max_fixations=int(args.max_fixations),
        base_dir=base_dir,
        excluded_subjects=tuple(args.exclude_subjects),
    )
    out_prop_csv = out_dir / f"fixation_viewing_time_share_by_position_long_max{int(args.max_fixations)}{tag_sfx}.csv"
    out_prop_png = fig_dir / f"fixation_viewing_time_share_by_position_max{int(args.max_fixations)}{tag_sfx}.png"
    out_prop_pdf = fig_dir / f"fixation_viewing_time_share_by_position_max{int(args.max_fixations)}{tag_sfx}.pdf"
    if prop is None or prop.empty:
        raise RuntimeError("No viewing-time share summary produced (empty after filtering?).")
    prop.to_csv(out_prop_csv, index=False)
    plot_time_share_summaries(prop, max_fixations=int(args.max_fixations), out_png=out_prop_png, out_pdf=out_prop_pdf)

    prop_7p = build_time_share_summaries_7plus(
        fix,
        max_fixations=int(args.max_fixations),
        base_dir=base_dir,
        excluded_subjects=tuple(args.exclude_subjects),
        keep_first_n=COLLAPSE_KEEP_FIRST_N,
    )
    if prop_7p is not None and (not prop_7p.empty):
        out_prop_7p_png = fig_dir / f"fixation_viewing_time_share_by_position_7plus_max{int(args.max_fixations)}{tag_sfx}.png"
        out_prop_7p_pdf = fig_dir / f"fixation_viewing_time_share_by_position_7plus_max{int(args.max_fixations)}{tag_sfx}.pdf"
        plot_time_share_summaries(
            prop_7p,
            max_fixations=int(COLLAPSE_KEEP_FIRST_N) + 1,
            out_png=out_prop_7p_png,
            out_pdf=out_prop_7p_pdf,
            xtick_labels=xticks_7plus,
        )

    print(f"Saved duration CSV: {out_csv}")
    print(f"Saved duration figure: {out_png}")
    print(f"Saved viewing-time share CSV: {out_prop_csv}")
    print(f"Saved viewing-time share figure: {out_prop_png}")
    if summary_7p is not None and (not summary_7p.empty):
        print(f"Saved duration figure (7+): {out_7p_png}")
    if prop_7p is not None and (not prop_7p.empty):
        print(f"Saved viewing-time share figure (7+): {out_prop_7p_png}")
    print(f"Loaded clean fixations: {in_path}")
    print(
        f"Subjects: {fix['subject_id'].nunique()} | "
        f"Trials: {fix[['subject_id','game','trial_number']].drop_duplicates().shape[0]} | "
        f"Fixations: {len(fix)}"
    )


if __name__ == "__main__":
    main()
