"""Visualize what early fixations target during choice (humans).

This script focuses on the first N fixations of each choice trial using the same
fixation ordering as `output/choice_fixations_clean_buffer_*.csv`.

For each fixation position (1..N), it summarizes the probability that a fixation
lands on:
  1) an offer-relevant item
  2) a top-ranked recalled-value item among the 6 items in that game
  3) a top-ranked recalled-value item among the 3 offer-relevant items for that option

Ranking mode:
- For the overall row, ranks use |recalled reward| (absolute magnitude).
- For take/leave rows, ranks use signed recalled reward (true value).

Recalled rewards are parsed from `data/<subid>/<subid>_valuerecall.csv`
using cue order from value-recall events in the main logfile.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_EXCLUDED = ("107", "131")


def _find_default_trial_metrics_csv(base_dir: Path) -> Optional[Path]:
    """Auto-detect a trial-metrics CSV produced by analysis/analyze_critical_items.py.

    Chooses the candidate with the most `is_*_trial` columns (i.e., most metrics).
    Prefers true-reward metrics when available.
    """

    fig_dir = base_dir / "figures"
    if not fig_dir.exists():
        return None

    def _pick_best(cands: List[Path]) -> Optional[Path]:
        best: Optional[Path] = None
        best_k = -1
        for p in cands:
            try:
                df = pd.read_csv(p, nrows=50)
            except Exception:
                continue
            cols = [c for c in df.columns if isinstance(c, str) and c.startswith("is_") and c.endswith("_trial")]
            k = len(cols)
            if k > best_k:
                best_k = k
                best = p
        return best

    true_cands = sorted(fig_dir.glob("critical_trials_human_true*.csv"))
    best_true = _pick_best(true_cands) if true_cands else None
    if best_true is not None:
        return best_true

    rec_cands = sorted(fig_dir.glob("critical_trials_human_recalled*.csv"))
    return _pick_best(rec_cands) if rec_cands else None


def load_trial_metrics(path: Path) -> pd.DataFrame:
    """Load per-trial metric flags from analyze_critical_items outputs."""

    df = pd.read_csv(path)
    if "subject" in df.columns and "subject_id" not in df.columns:
        df = df.rename(columns={"subject": "subject_id"})

    required = ["subject_id", "game", "trial_number"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Trial metrics CSV missing required columns: {missing}")

    df["subject_id"] = df["subject_id"].astype(str)
    df["game"] = pd.to_numeric(df["game"], errors="coerce")
    df["trial_number"] = pd.to_numeric(df["trial_number"], errors="coerce")
    df = df.dropna(subset=["game", "trial_number"]).copy()
    df["game"] = df["game"].astype(int)
    df["trial_number"] = df["trial_number"].astype(int)
    return df


def _infer_trial_metric_flag_columns(df: pd.DataFrame) -> List[str]:
    """Return binary `is_*_trial` columns to use for DO vs DONT trial splits."""

    candidates = [
        c
        for c in df.columns
        if isinstance(c, str) and c.startswith("is_") and c.endswith("_trial")
    ]

    out: List[str] = []
    for c in sorted(set(candidates)):
        v = pd.to_numeric(df[c], errors="coerce")
        u = set(v.dropna().astype(int).unique().tolist())
        if u.issubset({0, 1}):
            out.append(c)
    return out


def _subset_fixations_by_trial_metric(
    fix: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    metric_col: str,
    desired_value: int,
) -> pd.DataFrame:
    m = metrics[["subject_id", "game", "trial_number", metric_col]].copy()
    m[metric_col] = pd.to_numeric(m[metric_col], errors="coerce").fillna(0).astype(int)
    d = fix.merge(m, on=["subject_id", "game", "trial_number"], how="left")
    d[metric_col] = pd.to_numeric(d[metric_col], errors="coerce").fillna(0).astype(int)
    return d[d[metric_col] == int(desired_value)].copy()


def _sem(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    x = x.dropna()
    n = len(x)
    if n <= 1:
        return float("nan")
    return float(x.std(ddof=1) / np.sqrt(n))


def _mean_sem_agg(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    value_cols: Sequence[str],
) -> pd.DataFrame:
    """Aggregate mean + SEM across subjects for specified columns."""

    missing = [c for c in [*group_cols, *value_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"_mean_sem_agg missing required columns: {missing}")

    out = df.groupby(list(group_cols), as_index=False).agg(
        **{c: (c, "mean") for c in value_cols},
        n_subjects=("subject_id", lambda s: int(pd.Series(s).nunique())),
        **{f"{c}_sem": (c, _sem) for c in value_cols},
    )
    return out


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


def _parse_valence_token(token: object) -> Optional[int]:
    if token is None:
        return None
    if isinstance(token, float) and not np.isfinite(token):
        return None
    if isinstance(token, str):
        t = token.strip().lower()
        if t in {"+", "plus", "positive", "pos"}:
            return 1
        if t in {"-", "minus", "negative", "neg"}:
            return -1
    return None


def _parse_magnitude_token(token: object) -> Optional[int]:
    if token is None:
        return None
    if isinstance(token, (int, np.integer)):
        v = int(token)
        return v if 0 <= v <= 9 else None
    if isinstance(token, float) and np.isfinite(token):
        v = int(token)
        return v if 0 <= v <= 9 else None
    if isinstance(token, str):
        t = token.strip()
        if t == "":
            return None
        try:
            v = int(float(t))
        except Exception:
            return None
        return v if 0 <= v <= 9 else None
    return None


def _is_blank_vr_token(row: pd.Series) -> bool:
    tok = row.get("item")
    if tok is None:
        return True
    if isinstance(tok, float) and not np.isfinite(tok):
        return True
    if isinstance(tok, str) and tok.strip() == "":
        return True
    return False


def _extract_values_for_game(vr_rows: pd.DataFrame, n_items: int) -> List[float]:
    """Return exactly n_items recalled values from a game's valuerecall rows.

    Sequential parser that treats a single blank row as a missing-item placeholder.
    """

    values: List[float] = []
    i = 0
    vr_rows = vr_rows.reset_index(drop=True)

    while len(values) < n_items and i < len(vr_rows):
        row = vr_rows.iloc[i]

        if _is_blank_vr_token(row):
            values.append(np.nan)
            i += 1
            continue

        sign = _parse_valence_token(row.get("item"))
        mag = None
        if i + 1 < len(vr_rows):
            next_row = vr_rows.iloc[i + 1]
            if not _is_blank_vr_token(next_row):
                mag = _parse_magnitude_token(next_row.get("item"))
            i += 2
        else:
            i += 1

        if sign is None or mag is None:
            values.append(np.nan)
        else:
            values.append(float(sign * mag))

    while len(values) < n_items:
        values.append(np.nan)

    return values[:n_items]


def load_recalled_rewards(
    base_dir: Path,
    *,
    subjects: Optional[Sequence[str]] = None,
    excluded_subjects: Sequence[str] = DEFAULT_EXCLUDED,
) -> pd.DataFrame:
    """Return (subject_id, game, image) -> recalled reward."""

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
        vr_file = data_dir / subid / f"{subid}_valuerecall.csv"
        if not beh_file.exists():
            continue

        beh = pd.read_csv(beh_file)
        cues = beh[(beh.get("phase") == "memory") & (beh.get("event") == "value_recall")].copy()
        if cues.empty or (not {"game", "onset", "image"}.issubset(cues.columns)):
            continue
        cues = cues[["game", "onset", "image"]].dropna(subset=["game", "onset", "image"]).copy()
        cues["game"] = pd.to_numeric(cues["game"], errors="coerce")
        cues["onset"] = pd.to_numeric(cues["onset"], errors="coerce")
        cues = cues.dropna(subset=["game", "onset"]).copy()
        cues["game"] = cues["game"].astype(int)

        if vr_file.exists():
            vr = pd.read_csv(vr_file)
        else:
            # Preserve cue alignment even when transcription is missing; callers
            # may choose to fill missing recalls with true outcomes.
            vr = pd.DataFrame(columns=["game", "item", "onset"])

        if "game" in vr.columns:
            vr["game"] = pd.to_numeric(vr["game"], errors="coerce")
            vr = vr.dropna(subset=["game"]).copy()
            vr["game"] = vr["game"].astype(int)

        for game, cue_rows in cues.groupby("game"):
            cue_rows = cue_rows.sort_values("onset")
            items_ordered = cue_rows["image"].astype(str).tolist()
            if (not vr.empty) and ("game" in vr.columns):
                vr_g = vr[vr["game"] == int(game)].copy()
            else:
                vr_g = pd.DataFrame(columns=["item", "onset"])

            recalled_values = _extract_values_for_game(vr_g, n_items=len(items_ordered))
            out = pd.DataFrame(
                {
                    "subject_id": str(subid),
                    "game": int(game),
                    "image": items_ordered,
                    "reward_recalled": recalled_values,
                }
            )
            rows.append(out)

    if not rows:
        return pd.DataFrame(columns=["subject_id", "game", "image", "reward_recalled"])

    df = pd.concat(rows, ignore_index=True)
    df["subject_id"] = df["subject_id"].astype(str)
    df["game"] = pd.to_numeric(df["game"], errors="coerce").astype(int)
    df["reward_recalled"] = pd.to_numeric(df["reward_recalled"], errors="coerce")
    df = df.drop_duplicates(subset=["subject_id", "game", "image"], keep="first")
    return df


def load_true_rewards(
    base_dir: Path,
    *,
    subjects: Optional[Sequence[str]] = None,
    excluded_subjects: Sequence[str] = DEFAULT_EXCLUDED,
) -> pd.DataFrame:
    """Return (subject_id, game, image) -> true encoded reward."""

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
        enc = beh[(beh.get("phase") == "encoding") & (beh.get("event") == "image")].copy()
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
        out = enc.drop_duplicates(subset=["game", "image"], keep="first").copy()
        out["subject_id"] = str(subid)
        out = out.rename(columns={"outcome": "reward_true"})
        out = out[["subject_id", "game", "image", "reward_true"]]
        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=["subject_id", "game", "image", "reward_true"])

    df = pd.concat(rows, ignore_index=True)
    df["subject_id"] = df["subject_id"].astype(str)
    df["game"] = pd.to_numeric(df["game"], errors="coerce").astype(int)
    df["reward_true"] = pd.to_numeric(df["reward_true"], errors="coerce")
    df = df.drop_duplicates(subset=["subject_id", "game", "image"], keep="first")
    return df


def apply_missing_recall_fallback(
    recalled: pd.DataFrame,
    true_rewards: pd.DataFrame,
    *,
    strategy: str,
) -> pd.DataFrame:
    """Fill missing recalled rewards according to `strategy`.

    strategy:
      - 'true': replace NaN recalled values with the true encoded reward
      - 'lowest': leave NaN in place (missing will be ranked lowest downstream)
    """

    strategy = str(strategy).strip().lower()
    if strategy not in {"true", "lowest"}:
        raise ValueError("strategy must be one of {'true','lowest'}")
    if recalled.empty:
        return recalled
    if strategy == "lowest":
        return recalled

    merged = recalled.merge(true_rewards, on=["subject_id", "game", "image"], how="left")
    merged["reward_recalled"] = pd.to_numeric(merged["reward_recalled"], errors="coerce")
    merged["reward_true"] = pd.to_numeric(merged["reward_true"], errors="coerce")
    merged["reward_recalled"] = merged["reward_recalled"].fillna(merged["reward_true"])
    return merged.drop(columns=["reward_true"], errors="ignore")


def _load_reward_table_for_fixations(
    base_dir: Path,
    *,
    subjects: Sequence[str],
    excluded_subjects: Sequence[str],
    reward_source: str,
    missing_recall_fallback: str,
) -> pd.DataFrame:
    """Return (subject_id, game, image, reward_recalled) for downstream analyses.

    reward_source:
      - 'recalled': parse valuerecall transcripts via `load_recalled_rewards`
      - 'true': use the encoded reward from the main logfile encoding rows

    If reward_source='recalled' but no valuerecall-derived table can be built,
    this falls back to 'true' (with a console message) because NN simulation
    outputs do not include memory-phase value_recall events.
    """

    reward_source = str(reward_source).strip().lower()
    if reward_source not in {"recalled", "true"}:
        raise ValueError("reward_source must be one of {'recalled','true'}")

    subjects = [str(s) for s in subjects]
    excluded_subjects = [str(s) for s in excluded_subjects]

    if reward_source == "recalled":
        recalled = load_recalled_rewards(
            base_dir,
            subjects=subjects,
            excluded_subjects=tuple(excluded_subjects),
        )

        if recalled is None or recalled.empty or ("reward_recalled" not in recalled.columns):
            print(
                "No valuerecall-derived rewards found (missing memory/value_recall events?). "
                "Falling back to true encoded rewards."
            )
            reward_source = "true"
        else:
            missing_before = int(pd.to_numeric(recalled["reward_recalled"], errors="coerce").isna().sum())
            if str(missing_recall_fallback).strip().lower() == "true":
                true_rewards = load_true_rewards(
                    base_dir,
                    subjects=subjects,
                    excluded_subjects=tuple(excluded_subjects),
                )
                recalled = apply_missing_recall_fallback(recalled, true_rewards, strategy="true")
            missing_after = int(pd.to_numeric(recalled["reward_recalled"], errors="coerce").isna().sum())
            if missing_before or missing_after:
                print(
                    f"Missing recalled rewards: before={missing_before} after={missing_after} "
                    f"(fallback={str(missing_recall_fallback)})"
                )
            return recalled

    true_rewards = load_true_rewards(
        base_dir,
        subjects=subjects,
        excluded_subjects=tuple(excluded_subjects),
    )
    if true_rewards is None or true_rewards.empty:
        return pd.DataFrame(columns=["subject_id", "game", "image", "reward_recalled"])

    out = true_rewards.rename(columns={"reward_true": "reward_recalled"}).copy()
    out["reward_recalled"] = pd.to_numeric(out["reward_recalled"], errors="coerce")
    return out[["subject_id", "game", "image", "reward_recalled"]]


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
        "fixation_count",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Clean choice fixations missing required columns: {missing}")

    df["subject_id"] = df["subject_id"].astype(str)
    df["game"] = pd.to_numeric(df["game"], errors="coerce").astype(int)
    df["trial_number"] = pd.to_numeric(df["trial_number"], errors="coerce").astype(int)
    df["fixation_count"] = pd.to_numeric(df["fixation_count"], errors="coerce").astype(int)
    df["relevance"] = pd.to_numeric(df["relevance"], errors="coerce")

    df = df[df["image"].apply(_is_image_name)].copy()
    df = df[(df["choice"] == 1) | (df["choice"] == 2)].copy()
    return df


def _compute_rank_with_ties(values: pd.Series, *, n_bins: int, mode: str) -> pd.Series:
    """Rank values into 1..n_bins with ties assigned to the higher bin."""

    if mode not in {"abs", "signed"}:
        raise ValueError(f"Unknown rank mode: {mode!r}")

    if mode == "abs":
        v = values.abs().fillna(-1.0)
    else:
        v = values.fillna(-999.0)

    r = v.rank(method="max", ascending=True)
    r = r.clip(lower=1, upper=n_bins)
    return r.astype(int)


def _prepare_ranked_fixations(
    fix: pd.DataFrame,
    recalled: pd.DataFrame,
    *,
    max_fixations: int,
    rank_mode: str,
) -> pd.DataFrame:
    items = recalled.copy()
    if items.empty:
        raise RuntimeError("No recalled reward data found; check valuerecall files.")

    items["rank_6"] = items.groupby(["subject_id", "game"], sort=False)["reward_recalled"].transform(
        lambda s: _compute_rank_with_ties(s, n_bins=6, mode=rank_mode)
    )

    opts = fix[["subject_id", "game", "option"]].drop_duplicates()
    opt_items = opts.merge(items[["subject_id", "game", "image", "reward_recalled"]], on=["subject_id", "game"], how="left")
    opt_items["is_relevant_token"] = [
        int(isinstance(opt, str) and isinstance(img, str) and opt in img.split("_"))
        for opt, img in zip(opt_items["option"].tolist(), opt_items["image"].tolist())
    ]
    rel_items = opt_items[opt_items["is_relevant_token"] == 1].copy()
    rel_items["rank_rel_3"] = rel_items.groupby(["subject_id", "game", "option"], sort=False)["reward_recalled"].transform(
        lambda s: _compute_rank_with_ties(s, n_bins=3, mode=rank_mode)
    )
    rel_ranks = rel_items[["subject_id", "game", "option", "image", "rank_rel_3"]].drop_duplicates(
        subset=["subject_id", "game", "option", "image"]
    )

    irrel_items = opt_items[opt_items["is_relevant_token"] == 0].copy()
    irrel_items["rank_irrel_3"] = irrel_items.groupby(["subject_id", "game", "option"], sort=False)["reward_recalled"].transform(
        lambda s: _compute_rank_with_ties(s, n_bins=3, mode=rank_mode)
    )
    irrel_ranks = irrel_items[["subject_id", "game", "option", "image", "rank_irrel_3"]].drop_duplicates(
        subset=["subject_id", "game", "option", "image"]
    )

    df = fix.merge(items[["subject_id", "game", "image", "reward_recalled", "rank_6"]], on=["subject_id", "game", "image"], how="left")
    df = df.merge(rel_ranks, on=["subject_id", "game", "option", "image"], how="left")
    df = df.merge(irrel_ranks, on=["subject_id", "game", "option", "image"], how="left")

    df = df[df["fixation_count"] <= int(max_fixations)].copy()
    df["rank_6"] = pd.to_numeric(df["rank_6"], errors="coerce").fillna(1).astype(int)
    df["rank_rel_3"] = pd.to_numeric(df["rank_rel_3"], errors="coerce")
    df["rank_irrel_3"] = pd.to_numeric(df["rank_irrel_3"], errors="coerce")
    df["is_relevant"] = (pd.to_numeric(df["relevance"], errors="coerce").fillna(0.0) > 0.5).astype(int)
    df["is_irrelevant"] = (df["is_relevant"] == 0).astype(int)
    df["rank_mode"] = str(rank_mode)
    return df


def build_fixation_position_summary(
    fix: pd.DataFrame,
    recalled: pd.DataFrame,
    *,
    max_fixations: int,
    rank_mode: str,
) -> pd.DataFrame:
    df = _prepare_ranked_fixations(fix, recalled, max_fixations=max_fixations, rank_mode=rank_mode)

    df["is_top_6"] = (df["rank_6"] == 6).astype(int)
    df["is_top_rel_3"] = ((df["is_relevant"] == 1) & (pd.to_numeric(df["rank_rel_3"], errors="coerce") == 3)).astype(int)
    df["is_top_irrel_3"] = ((df["is_irrelevant"] == 1) & (pd.to_numeric(df["rank_irrel_3"], errors="coerce") == 3)).astype(int)

    # Per-subject proportions, then aggregate across subjects (equal weight per subject).
    per_sub = (
        df.groupby(["subject_id", "fixation_count"], as_index=False)
        .agg(
            n_fixations=("image", "size"),
            p_relevant=("is_relevant", "mean"),
            p_top_rank_6=("is_top_6", "mean"),
            p_top_rank_rel_3=("is_top_rel_3", "mean"),
            p_top_rank_irrel_3=("is_top_irrel_3", "mean"),
            n_missing_recalled=(
                "reward_recalled",
                lambda s: int(pd.isna(pd.to_numeric(s, errors="coerce")).sum()),
            ),
        )
        .rename(columns={"fixation_count": "fixation_position"})
    )

    out = _mean_sem_agg(
        per_sub,
        group_cols=["fixation_position"],
        value_cols=["p_relevant", "p_top_rank_6", "p_top_rank_rel_3", "p_top_rank_irrel_3"],
    )
    # Totals for reference (not used for mean/SEM).
    totals = per_sub.groupby("fixation_position", as_index=False).agg(
        n_fixations=("n_fixations", "sum"),
        n_missing_recalled=("n_missing_recalled", "sum"),
    )
    out = out.merge(totals, on="fixation_position", how="left")
    out["max_fixations"] = int(max_fixations)
    out["rank_mode"] = str(rank_mode)
    return out


def build_rank_distributions(
    fix: pd.DataFrame,
    recalled: pd.DataFrame,
    *,
    max_fixations: int,
    rank_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = _prepare_ranked_fixations(fix, recalled, max_fixations=max_fixations, rank_mode=rank_mode)

    subjects = sorted(df["subject_id"].dropna().astype(str).unique().tolist())
    pos = pd.DataFrame({"fixation_position": list(range(1, int(max_fixations) + 1))})

    # Rank(6): per-subject distribution over bins at each position, then mean+SEM across subjects.
    bins6 = pd.DataFrame({"rank_bin": list(range(1, 7))})
    grid6 = (
        pd.DataFrame({"subject_id": subjects})
        .merge(pos, how="cross")
        .merge(bins6, how="cross")
    )
    counts6 = (
        df.groupby(["subject_id", "fixation_count", "rank_6"], as_index=False)
        .size()
        .rename(columns={"fixation_count": "fixation_position", "rank_6": "rank_bin", "size": "n"})
    )
    totals6 = (
        df.groupby(["subject_id", "fixation_count"], as_index=False)
        .size()
        .rename(columns={"fixation_count": "fixation_position", "size": "n_total"})
    )
    per6 = grid6.merge(counts6, on=["subject_id", "fixation_position", "rank_bin"], how="left")
    per6 = per6.merge(totals6, on=["subject_id", "fixation_position"], how="left")
    per6["n"] = per6["n"].fillna(0).astype(int)
    per6["p"] = np.where(per6["n_total"].fillna(0).to_numpy(dtype=float) > 0, per6["n"] / per6["n_total"], np.nan)

    rank6 = _mean_sem_agg(per6, group_cols=["fixation_position", "rank_bin"], value_cols=["p"])
    totals6_all = per6.groupby(["fixation_position", "rank_bin"], as_index=False).agg(n=("n", "sum"))
    totals6_pos = per6.groupby(["fixation_position"], as_index=False).agg(n_total=("n_total", "sum"))
    rank6 = rank6.merge(totals6_all, on=["fixation_position", "rank_bin"], how="left")
    rank6 = rank6.merge(totals6_pos, on="fixation_position", how="left")
    rank6["n"] = rank6["n"].fillna(0).astype(int)
    rank6["n_total"] = rank6["n_total"].fillna(0).astype(int)
    rank6["rank_mode"] = str(rank_mode)

    rel = df[(df["is_relevant"] == 1) & (pd.to_numeric(df["rank_rel_3"], errors="coerce").notna())].copy()
    rel["rank_rel_3"] = pd.to_numeric(rel["rank_rel_3"], errors="coerce").astype(int)

    bins3 = pd.DataFrame({"rank_bin": list(range(1, 4))})
    grid3 = (
        pd.DataFrame({"subject_id": subjects})
        .merge(pos, how="cross")
        .merge(bins3, how="cross")
    )

    if rel.empty:
        rel3 = pd.DataFrame(columns=["fixation_position", "rank_bin", "p", "p_sem", "n_subjects", "n", "n_relevant", "rank_mode"])
    else:
        counts3 = (
            rel.groupby(["subject_id", "fixation_count", "rank_rel_3"], as_index=False)
            .size()
            .rename(columns={"fixation_count": "fixation_position", "rank_rel_3": "rank_bin", "size": "n"})
        )
        totals3 = (
            rel.groupby(["subject_id", "fixation_count"], as_index=False)
            .size()
            .rename(columns={"fixation_count": "fixation_position", "size": "n_relevant"})
        )
        per3 = grid3.merge(counts3, on=["subject_id", "fixation_position", "rank_bin"], how="left")
        per3 = per3.merge(totals3, on=["subject_id", "fixation_position"], how="left")
        per3["n"] = per3["n"].fillna(0).astype(int)
        per3["p"] = np.where(per3["n_relevant"].fillna(0).to_numpy(dtype=float) > 0, per3["n"] / per3["n_relevant"], np.nan)

        rel3 = _mean_sem_agg(per3, group_cols=["fixation_position", "rank_bin"], value_cols=["p"])
        totals3_all = per3.groupby(["fixation_position", "rank_bin"], as_index=False).agg(n=("n", "sum"))
        totals3_pos = per3.groupby(["fixation_position"], as_index=False).agg(n_relevant=("n_relevant", "sum"))
        rel3 = rel3.merge(totals3_all, on=["fixation_position", "rank_bin"], how="left")
        rel3 = rel3.merge(totals3_pos, on="fixation_position", how="left")
        rel3["n"] = rel3["n"].fillna(0).astype(int)
        rel3["n_relevant"] = rel3["n_relevant"].fillna(0).astype(int)
        rel3["rank_mode"] = str(rank_mode)

        # Irrelevant-only distribution over 3-bin rank, conditional on being irrelevant at that position.
        irrel = df[(df["is_irrelevant"] == 1) & (pd.to_numeric(df["rank_irrel_3"], errors="coerce").notna())].copy()
        irrel["rank_irrel_3"] = pd.to_numeric(irrel["rank_irrel_3"], errors="coerce").astype(int)

        if irrel.empty:
            irrel3 = pd.DataFrame(columns=["fixation_position", "rank_bin", "p", "p_sem", "n_subjects", "n", "n_irrelevant", "rank_mode"])
        else:
            countsi = (
                irrel.groupby(["subject_id", "fixation_count", "rank_irrel_3"], as_index=False)
                .size()
                .rename(columns={"fixation_count": "fixation_position", "rank_irrel_3": "rank_bin", "size": "n"})
            )
            totalsi = (
                irrel.groupby(["subject_id", "fixation_count"], as_index=False)
                .size()
                .rename(columns={"fixation_count": "fixation_position", "size": "n_irrelevant"})
            )
            peri = grid3.merge(countsi, on=["subject_id", "fixation_position", "rank_bin"], how="left")
            peri = peri.merge(totalsi, on=["subject_id", "fixation_position"], how="left")
            peri["n"] = peri["n"].fillna(0).astype(int)
            peri["p"] = np.where(peri["n_irrelevant"].fillna(0).to_numpy(dtype=float) > 0, peri["n"] / peri["n_irrelevant"], np.nan)

            irrel3 = _mean_sem_agg(peri, group_cols=["fixation_position", "rank_bin"], value_cols=["p"])
            total_i_all = peri.groupby(["fixation_position", "rank_bin"], as_index=False).agg(n=("n", "sum"))
            total_i_pos = peri.groupby(["fixation_position"], as_index=False).agg(n_irrelevant=("n_irrelevant", "sum"))
            irrel3 = irrel3.merge(total_i_all, on=["fixation_position", "rank_bin"], how="left")
            irrel3 = irrel3.merge(total_i_pos, on="fixation_position", how="left")
            irrel3["n"] = irrel3["n"].fillna(0).astype(int)
            irrel3["n_irrelevant"] = irrel3["n_irrelevant"].fillna(0).astype(int)
            irrel3["rank_mode"] = str(rank_mode)

        return rank6, rel3, irrel3


def compute_all_summaries(
    fix: pd.DataFrame,
    recalled: pd.DataFrame,
    *,
    max_fixations: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subsets = [
        ("overall", fix, "abs"),
        ("take", fix[fix["choice"] == 1].copy(), "signed"),
        ("leave", fix[fix["choice"] == 2].copy(), "signed"),
    ]

    sum_frames: List[pd.DataFrame] = []
    rank6_frames: List[pd.DataFrame] = []
    rel3_frames: List[pd.DataFrame] = []
    irrel3_frames: List[pd.DataFrame] = []

    for label, sub, rank_mode in subsets:
        if sub.empty:
            continue
        s = build_fixation_position_summary(sub, recalled, max_fixations=int(max_fixations), rank_mode=rank_mode)
        r6, r3, ir3 = build_rank_distributions(sub, recalled, max_fixations=int(max_fixations), rank_mode=rank_mode)

        s["choice_split"] = label
        r6["choice_split"] = label
        r3["choice_split"] = label
        ir3["choice_split"] = label

        sum_frames.append(s)
        rank6_frames.append(r6)
        rel3_frames.append(r3)
        irrel3_frames.append(ir3)

    summary = pd.concat(sum_frames, ignore_index=True) if sum_frames else pd.DataFrame()
    rank6_long = pd.concat(rank6_frames, ignore_index=True) if rank6_frames else pd.DataFrame()
    rel3_long = pd.concat(rel3_frames, ignore_index=True) if rel3_frames else pd.DataFrame()
    irrel3_long = pd.concat(irrel3_frames, ignore_index=True) if irrel3_frames else pd.DataFrame()
    return summary, rank6_long, rel3_long, irrel3_long


def build_sign_summary(
    fix: pd.DataFrame,
    recalled: pd.DataFrame,
    *,
    max_fixations: int,
    subset: str,
) -> pd.DataFrame:
    """By fixation position, summarize P(positive) vs P(negative) recalled value.

    Parameters
    ----------
    subset:
        'overall' | 'relevant' | 'irrelevant'
        Probabilities are always computed **out of all fixations** at a given
        fixation position. For 'relevant' and 'irrelevant', this means the
        returned values are *joint* probabilities, e.g.
        $P(\\text{positive} \\wedge \\text{relevant} \\mid \\text{position})$.
    """

    if subset not in {"overall", "relevant", "irrelevant"}:
        raise ValueError(f"Unknown sign subset: {subset!r}")

    df = _prepare_ranked_fixations(fix, recalled, max_fixations=int(max_fixations), rank_mode="signed")

    if subset == "overall":
        mask = pd.Series(True, index=df.index)
    elif subset == "relevant":
        mask = df["is_relevant"] == 1
    else:
        mask = df["is_irrelevant"] == 1
    rr = pd.to_numeric(df["reward_recalled"], errors="coerce")
    finite = np.isfinite(rr.to_numpy(dtype=float))
    df["is_pos"] = ((rr > 0) & mask).astype(int)
    df["is_neg"] = ((rr < 0) & mask).astype(int)
    df["is_missing"] = ((~finite) & mask).astype(int)

    per_sub = (
        df.groupby(["subject_id", "fixation_count"], as_index=False)
        .agg(
            n_fixations=("image", "size"),
            p_pos=("is_pos", "mean"),
            p_neg=("is_neg", "mean"),
            p_missing=("is_missing", "mean"),
        )
        .rename(columns={"fixation_count": "fixation_position"})
    )

    out = _mean_sem_agg(per_sub, group_cols=["fixation_position"], value_cols=["p_pos", "p_neg", "p_missing"])
    totals = per_sub.groupby("fixation_position", as_index=False).agg(n_fixations=("n_fixations", "sum"))
    out = out.merge(totals, on="fixation_position", how="left")
    out["max_fixations"] = int(max_fixations)
    out["relevance_subset"] = str(subset)
    return out


def plot_sign_summary(sign_summary: pd.DataFrame, *, out_png: Path, out_pdf: Path) -> None:
    choice_order = ["overall", "take", "leave"]
    subset_order = ["overall", "relevant", "irrelevant"]

    def _set_prob_ylim(ax: plt.Axes, y_values: Iterable[float]) -> None:
        vals = [float(v) for v in y_values if v is not None and np.isfinite(v)]
        if not vals:
            ax.set_ylim(0.0, 1.0)
            return
        ymax = max(vals)
        upper = min(1.0, max(0.05, ymax * 1.10, ymax + 0.02))
        ax.set_ylim(0.0, upper)

    fig, axes = plt.subplots(3, 3, figsize=(14, 10.5), constrained_layout=True)
    for row_i, choice_label in enumerate(choice_order):
        for col_j, subset_label in enumerate(subset_order):
            ax = axes[row_i, col_j]
            s = sign_summary[
                (sign_summary.get("choice_split") == choice_label)
                & (sign_summary.get("relevance_subset") == subset_label)
            ].copy()
            if s.empty:
                ax.set_title(f"{choice_label} / {subset_label}: (no trials)")
                ax.axis("off")
                continue
            x = s["fixation_position"].to_numpy(dtype=int)

            def _plot_mean_sem(y_col: str, label: str) -> None:
                y = pd.to_numeric(s[y_col], errors="coerce").to_numpy(dtype=float)
                sem_col = f"{y_col}_sem"
                ax.plot(x, y, marker="o", label=label)
                if sem_col in s.columns:
                    se = pd.to_numeric(s[sem_col], errors="coerce").to_numpy(dtype=float)
                    lo = np.clip(y - se, 0.0, 1.0)
                    hi = np.clip(y + se, 0.0, 1.0)
                    ax.fill_between(x, lo, hi, alpha=0.20)

            _plot_mean_sem("p_pos", "P(positive recalled value)")
            _plot_mean_sem("p_neg", "P(negative recalled value)")
            ax.set_title(f"{choice_label}: {subset_label}")
            ax.set_xlabel("Fixation position")
            ax.set_ylabel("Probability")
            y_for_ylim = [
                pd.to_numeric(s.get("p_pos"), errors="coerce"),
                pd.to_numeric(s.get("p_neg"), errors="coerce"),
                pd.to_numeric(s.get("p_pos_sem"), errors="coerce"),
                pd.to_numeric(s.get("p_neg_sem"), errors="coerce"),
            ]
            y_for_ylim = pd.concat([v for v in y_for_ylim if isinstance(v, pd.Series)]).dropna().to_numpy(dtype=float)
            if "p_pos_sem" in s.columns and "p_neg_sem" in s.columns:
                # Consider mean+SEM in ylim.
                y_for_ylim = np.concatenate(
                    [
                        y_for_ylim,
                        np.clip((pd.to_numeric(s["p_pos"], errors="coerce") + pd.to_numeric(s["p_pos_sem"], errors="coerce")).to_numpy(dtype=float), 0.0, 1.0),
                        np.clip((pd.to_numeric(s["p_neg"], errors="coerce") + pd.to_numeric(s["p_neg_sem"], errors="coerce")).to_numpy(dtype=float), 0.0, 1.0),
                    ]
                )
            _set_prob_ylim(ax, y_for_ylim)
            ax.grid(True, alpha=0.3)
            if row_i == 0 and col_j == 0:
                ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Early-fixation targeting by recalled value sign (overall vs relevant vs irrelevant)")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def build_absrank6_by_sign_summary(
    fix: pd.DataFrame,
    recalled: pd.DataFrame,
    *,
    max_fixations: int,
    subset: str,
) -> pd.DataFrame:
    """Mean abs-rank6 by fixation position, split by value sign.

    - Rank bins are computed using absolute magnitude (rank_mode='abs') within (subject, game).
    - Sign is based on the signed recalled reward (after any missing-recall fallback).
    - Means are computed per subject first, then averaged across subjects with SEM.

    Parameters
    ----------
    subset:
        'overall' | 'relevant' | 'irrelevant'
        Here, subsetting is applied *before* splitting by sign.
    """

    subset = str(subset).strip().lower()
    if subset not in {"overall", "relevant", "irrelevant"}:
        raise ValueError(f"Unknown subset: {subset!r}")

    df = _prepare_ranked_fixations(fix, recalled, max_fixations=int(max_fixations), rank_mode="abs")
    if df.empty:
        return pd.DataFrame()

    if subset == "relevant":
        df = df[df["is_relevant"] == 1].copy()
    elif subset == "irrelevant":
        df = df[df["is_irrelevant"] == 1].copy()

    rr = pd.to_numeric(df["reward_recalled"], errors="coerce")
    df = df[np.isfinite(rr.to_numpy(dtype=float))].copy()
    if df.empty:
        return pd.DataFrame()

    rr = pd.to_numeric(df["reward_recalled"], errors="coerce")
    df["value_sign"] = np.where(rr.to_numpy(dtype=float) > 0, "pos", "neg")
    df = df[rr != 0].copy()
    if df.empty:
        return pd.DataFrame()

    per_sub = (
        df.groupby(["subject_id", "fixation_count", "value_sign"], as_index=False)
        .agg(
            n_fixations=("rank_6", "size"),
            mean_absrank6=("rank_6", "mean"),
        )
        .rename(columns={"fixation_count": "fixation_position"})
    )

    out = _mean_sem_agg(per_sub, group_cols=["fixation_position", "value_sign"], value_cols=["mean_absrank6"])
    totals = per_sub.groupby(["fixation_position", "value_sign"], as_index=False).agg(n_fixations=("n_fixations", "sum"))
    out = out.merge(totals, on=["fixation_position", "value_sign"], how="left")
    out["max_fixations"] = int(max_fixations)
    out["relevance_subset"] = subset
    return out


def plot_absrank6_by_sign_summary(df: pd.DataFrame, *, out_png: Path, out_pdf: Path) -> None:
    """3x3 plot: mean abs-rank6 (pos vs neg) across fixation positions."""

    if df is None or df.empty:
        return

    choice_order = ["overall", "take", "leave"]
    subset_order = ["overall", "relevant", "irrelevant"]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10.5), constrained_layout=True)
    for row_i, choice_label in enumerate(choice_order):
        for col_j, subset_label in enumerate(subset_order):
            ax = axes[row_i, col_j]
            s = df[(df.get("choice_split") == choice_label) & (df.get("relevance_subset") == subset_label)].copy()
            if s.empty:
                ax.set_title(f"{choice_label} / {subset_label}: (no data)")
                ax.axis("off")
                continue

            s["fixation_position"] = pd.to_numeric(s["fixation_position"], errors="coerce")
            s = s.dropna(subset=["fixation_position"]).copy()
            s["fixation_position"] = s["fixation_position"].astype(int)

            def _plot_sign(sign: str, *, color: str, label: str) -> None:
                ss = s[s.get("value_sign") == sign].copy()
                if ss.empty:
                    return
                ss = ss.sort_values("fixation_position")
                x = ss["fixation_position"].to_numpy(dtype=int)
                y = pd.to_numeric(ss["mean_absrank6"], errors="coerce").to_numpy(dtype=float)
                ax.plot(x, y, marker="o", linewidth=1.5, color=color, label=label)
                sem_col = "mean_absrank6_sem"
                if sem_col in ss.columns:
                    se = pd.to_numeric(ss[sem_col], errors="coerce").to_numpy(dtype=float)
                    lo = np.clip(y - se, 1.0, 6.0)
                    hi = np.clip(y + se, 1.0, 6.0)
                    ax.fill_between(x, lo, hi, color=color, alpha=0.15)

            _plot_sign("pos", color="tab:blue", label="positive")
            _plot_sign("neg", color="tab:red", label="negative")

            ax.set_title(f"{choice_label}: {subset_label}")
            ax.set_xlabel("Fixation position")
            ax.set_ylabel("Mean |value| rank (1–6)")
            ax.set_ylim(1.0, 6.0)
            ax.grid(True, alpha=0.3)
            if row_i == 0 and col_j == 0:
                ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Early-fixation targeting by sign + magnitude (mean |value|-rank)")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def build_absrank6_dist_by_sign_joint(
    fix: pd.DataFrame,
    recalled: pd.DataFrame,
    *,
    max_fixations: int,
    subset: str,
) -> pd.DataFrame:
    r"""Joint distribution of (sign x abs-rank6) out of all fixations.

    For each fixation position, returns joint probabilities like:
    $P(\mathrm{pos} \wedge \mathrm{absrank}=b \wedge \mathrm{subset} \mid \mathrm{position})$.

    - abs-rank6 is computed within (subject, game) based on |recalled reward|.
    - sign is based on signed recalled reward.
    - Denominator is **all fixations at that position** (not conditional on subset),
      matching the convention used in `build_sign_summary`.
    """

    subset = str(subset).strip().lower()
    if subset not in {"overall", "relevant", "irrelevant"}:
        raise ValueError(f"Unknown subset: {subset!r}")

    df = _prepare_ranked_fixations(fix, recalled, max_fixations=int(max_fixations), rank_mode="abs")
    if df.empty:
        return pd.DataFrame()

    if subset == "overall":
        mask = pd.Series(True, index=df.index)
    elif subset == "relevant":
        mask = df["is_relevant"] == 1
    else:
        mask = df["is_irrelevant"] == 1

    df["fixation_position"] = pd.to_numeric(df["fixation_count"], errors="coerce").astype(int)
    rr = pd.to_numeric(df["reward_recalled"], errors="coerce")
    finite = np.isfinite(rr.to_numpy(dtype=float))

    # Only positive/negative are used for sign-splitting; zeros/missing are excluded
    # from the numerator but still present in the denominator (all fixations).
    df["value_sign"] = np.where(rr.to_numpy(dtype=float) > 0, "pos", np.where(rr.to_numpy(dtype=float) < 0, "neg", "other"))

    n_total = (
        df.groupby(["subject_id", "fixation_position"], as_index=False)
        .agg(n_total=("image", "size"))
        .copy()
    )

    df_num = df[mask & finite & (df["value_sign"].isin(["pos", "neg"]))].copy()
    if df_num.empty:
        return pd.DataFrame()

    counts = (
        df_num.groupby(["subject_id", "fixation_position", "value_sign", "rank_6"], as_index=False)
        .size()
        .rename(columns={"size": "n", "rank_6": "rank_bin"})
    )

    # Expand to include zero-probability combinations for each subject/position.
    signs = pd.DataFrame({"value_sign": ["pos", "neg"]})
    bins = pd.DataFrame({"rank_bin": [1, 2, 3, 4, 5, 6]})
    base = n_total.merge(signs, how="cross").merge(bins, how="cross")
    per_sub = base.merge(counts, on=["subject_id", "fixation_position", "value_sign", "rank_bin"], how="left")
    per_sub["n"] = pd.to_numeric(per_sub["n"], errors="coerce").fillna(0.0)
    per_sub["p"] = (per_sub["n"] / per_sub["n_total"]).astype(float)

    out = _mean_sem_agg(per_sub, group_cols=["fixation_position", "value_sign", "rank_bin"], value_cols=["p"])

    totals = per_sub.groupby(["fixation_position", "value_sign", "rank_bin"], as_index=False).agg(
        n_fixations=("n", "sum"),
    )
    out = out.merge(totals, on=["fixation_position", "value_sign", "rank_bin"], how="left")
    out["max_fixations"] = int(max_fixations)
    out["relevance_subset"] = subset
    return out


def plot_absrank6_dist_by_sign_joint(df: pd.DataFrame, *, out_png: Path, out_pdf: Path) -> None:
    """3x3 plot: P(sign & absrank bin | position), with SEM bands."""

    if df is None or df.empty:
        return

    choice_order = ["overall", "take", "leave"]
    subset_order = ["overall", "relevant", "irrelevant"]

    def _set_prob_ylim(ax: plt.Axes, y_values: Iterable[float]) -> None:
        vals = [float(v) for v in y_values if v is not None and np.isfinite(v)]
        if not vals:
            ax.set_ylim(0.0, 1.0)
            return
        ymax = max(vals)
        upper = min(1.0, max(0.05, ymax * 1.10, ymax + 0.02))
        ax.set_ylim(0.0, upper)

    # Colors encode rank bin; linestyle encodes sign.
    bin_colors = {1: "tab:gray", 2: "tab:purple", 3: "tab:blue", 4: "tab:green", 5: "tab:orange", 6: "tab:red"}
    sign_styles = {"pos": "-", "neg": "--"}

    fig, axes = plt.subplots(3, 3, figsize=(14, 10.5), constrained_layout=True)
    for row_i, choice_label in enumerate(choice_order):
        for col_j, subset_label in enumerate(subset_order):
            ax = axes[row_i, col_j]
            s = df[(df.get("choice_split") == choice_label) & (df.get("relevance_subset") == subset_label)].copy()
            if s.empty:
                ax.set_title(f"{choice_label} / {subset_label}: (no data)")
                ax.axis("off")
                continue

            s["fixation_position"] = pd.to_numeric(s["fixation_position"], errors="coerce")
            s["rank_bin"] = pd.to_numeric(s["rank_bin"], errors="coerce")
            s = s.dropna(subset=["fixation_position", "rank_bin"]).copy()
            s["fixation_position"] = s["fixation_position"].astype(int)
            s["rank_bin"] = s["rank_bin"].astype(int)

            x_vals = sorted(s["fixation_position"].unique().tolist())

            y_for_ylim: List[float] = []
            for sign in ["pos", "neg"]:
                for b in range(1, 7):
                    ss = s[(s["value_sign"] == sign) & (s["rank_bin"] == b)].copy()
                    if ss.empty:
                        continue
                    ss = ss.sort_values("fixation_position")
                    x = ss["fixation_position"].to_numpy(dtype=int)
                    y = pd.to_numeric(ss["p"], errors="coerce").to_numpy(dtype=float)
                    col = bin_colors.get(b, "tab:gray")
                    ls = sign_styles.get(sign, "-")
                    ax.plot(x, y, marker="o", linewidth=1.2, color=col, linestyle=ls)
                    y_for_ylim.extend([float(v) for v in y if np.isfinite(v)])
                    if "p_sem" in ss.columns:
                        se = pd.to_numeric(ss["p_sem"], errors="coerce").to_numpy(dtype=float)
                        lo = np.clip(y - se, 0.0, 1.0)
                        hi = np.clip(y + se, 0.0, 1.0)
                        ax.fill_between(x, lo, hi, color=col, alpha=0.12)
                        y_for_ylim.extend([float(v) for v in hi if np.isfinite(v)])

            ax.set_title(f"{choice_label}: {subset_label}")
            ax.set_xlabel("Fixation position")
            ax.set_ylabel("P(sign & |value|-rank bin | pos)")
            _set_prob_ylim(ax, y_for_ylim)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(x_vals)

            if row_i == 0 and col_j == 0:
                from matplotlib.lines import Line2D

                bin_handles = [
                    Line2D([0], [0], color=bin_colors[b], lw=2, label=f"bin {b}")
                    for b in range(1, 7)
                ]
                sign_handles = [
                    Line2D([0], [0], color="black", lw=2, linestyle=sign_styles["pos"], label="positive"),
                    Line2D([0], [0], color="black", lw=2, linestyle=sign_styles["neg"], label="negative"),
                ]
                ax.legend(handles=sign_handles + bin_handles, frameon=False, fontsize=8, ncol=2)

    fig.suptitle("Early-fixation targeting by sign + magnitude (P(sign & |value|-rank bin | position))")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def build_absrank3_within_subset_dist_by_sign_joint(
    fix: pd.DataFrame,
    recalled: pd.DataFrame,
    *,
    max_fixations: int,
    subset: str,
) -> pd.DataFrame:
    r"""Joint distribution of (sign x within-subset abs-rank3) out of all fixations.

    This differs from `build_absrank6_dist_by_sign_joint` in that the rank bins are
    computed **within the 3 relevant items** (or **within the 3 irrelevant items**)
    for a given (subject, game, option). Hence rank bins are 1..3.

    For each fixation position, returns joint probabilities like:
    $P(\mathrm{pos} \wedge \mathrm{absrank}_{\mathrm{subset}}=b \wedge \mathrm{subset} \mid \mathrm{position})$.

    - within-subset abs-rank3 is computed based on |recalled reward|.
    - sign is based on signed recalled reward.
    - Denominator is **all fixations at that position** (not conditional on subset).
    """

    subset = str(subset).strip().lower()
    if subset not in {"relevant", "irrelevant"}:
        raise ValueError(f"subset must be one of {{'relevant','irrelevant'}}, got {subset!r}")

    df = _prepare_ranked_fixations(fix, recalled, max_fixations=int(max_fixations), rank_mode="abs")
    if df.empty:
        return pd.DataFrame()

    if subset == "relevant":
        mask = df["is_relevant"] == 1
        rank_col = "rank_rel_3"
    else:
        mask = df["is_irrelevant"] == 1
        rank_col = "rank_irrel_3"

    df["fixation_position"] = pd.to_numeric(df["fixation_count"], errors="coerce").astype(int)
    rr = pd.to_numeric(df["reward_recalled"], errors="coerce")
    finite = np.isfinite(rr.to_numpy(dtype=float))

    # Only positive/negative are used for sign-splitting; zeros/missing are excluded
    # from the numerator but still present in the denominator (all fixations).
    df["value_sign"] = np.where(
        rr.to_numpy(dtype=float) > 0,
        "pos",
        np.where(rr.to_numpy(dtype=float) < 0, "neg", "other"),
    )

    n_total = (
        df.groupby(["subject_id", "fixation_position"], as_index=False)
        .agg(n_total=("image", "size"))
        .copy()
    )

    rank_vals = pd.to_numeric(df[rank_col], errors="coerce")
    df_num = df[mask & finite & (df["value_sign"].isin(["pos", "neg"])) & rank_vals.notna()].copy()
    if df_num.empty:
        return pd.DataFrame()

    df_num["rank_bin"] = pd.to_numeric(df_num[rank_col], errors="coerce").astype(int)
    df_num = df_num[df_num["rank_bin"].isin([1, 2, 3])].copy()
    if df_num.empty:
        return pd.DataFrame()

    counts = (
        df_num.groupby(["subject_id", "fixation_position", "value_sign", "rank_bin"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )

    # Expand to include zero-probability combinations for each subject/position.
    signs = pd.DataFrame({"value_sign": ["pos", "neg"]})
    bins = pd.DataFrame({"rank_bin": [1, 2, 3]})
    base = n_total.merge(signs, how="cross").merge(bins, how="cross")
    per_sub = base.merge(counts, on=["subject_id", "fixation_position", "value_sign", "rank_bin"], how="left")
    per_sub["n"] = pd.to_numeric(per_sub["n"], errors="coerce").fillna(0.0)
    per_sub["p"] = (per_sub["n"] / per_sub["n_total"]).astype(float)

    out = _mean_sem_agg(per_sub, group_cols=["fixation_position", "value_sign", "rank_bin"], value_cols=["p"])
    totals = per_sub.groupby(["fixation_position", "value_sign", "rank_bin"], as_index=False).agg(
        n_fixations=("n", "sum"),
    )
    out = out.merge(totals, on=["fixation_position", "value_sign", "rank_bin"], how="left")
    out["max_fixations"] = int(max_fixations)
    out["relevance_subset"] = subset
    return out


def plot_absrank3_within_subset_dist_by_sign_joint(df: pd.DataFrame, *, out_png: Path, out_pdf: Path) -> None:
    """2x3 plot: P(sign & within-subset absrank3 bin | position), with SEM bands."""

    if df is None or df.empty:
        return

    choice_order = ["overall", "take", "leave"]
    subset_order = ["relevant", "irrelevant"]

    def _set_prob_ylim(ax: plt.Axes, y_values: Iterable[float]) -> None:
        vals = [float(v) for v in y_values if v is not None and np.isfinite(v)]
        if not vals:
            ax.set_ylim(0.0, 1.0)
            return
        ymax = max(vals)
        upper = min(1.0, max(0.05, ymax * 1.10, ymax + 0.02))
        ax.set_ylim(0.0, upper)

    # Colors encode rank bin; linestyle encodes sign.
    bin_colors = {1: "tab:gray", 2: "tab:blue", 3: "tab:red"}
    sign_styles = {"pos": "-", "neg": "--"}

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=True)
    for row_i, subset_label in enumerate(subset_order):
        for col_j, choice_label in enumerate(choice_order):
            ax = axes[row_i, col_j]
            s = df[(df.get("choice_split") == choice_label) & (df.get("relevance_subset") == subset_label)].copy()
            if s.empty:
                ax.set_title(f"{choice_label} / {subset_label}: (no data)")
                ax.axis("off")
                continue

            s["fixation_position"] = pd.to_numeric(s["fixation_position"], errors="coerce")
            s["rank_bin"] = pd.to_numeric(s["rank_bin"], errors="coerce")
            s = s.dropna(subset=["fixation_position", "rank_bin"]).copy()
            s["fixation_position"] = s["fixation_position"].astype(int)
            s["rank_bin"] = s["rank_bin"].astype(int)

            x_vals = sorted(s["fixation_position"].unique().tolist())
            y_for_ylim: List[float] = []
            for sign in ["pos", "neg"]:
                for b in [1, 2, 3]:
                    ss = s[(s["value_sign"] == sign) & (s["rank_bin"] == b)].copy()
                    if ss.empty:
                        continue
                    ss = ss.sort_values("fixation_position")
                    x = ss["fixation_position"].to_numpy(dtype=int)
                    y = pd.to_numeric(ss["p"], errors="coerce").to_numpy(dtype=float)
                    col = bin_colors.get(b, "tab:gray")
                    ls = sign_styles.get(sign, "-")
                    ax.plot(x, y, marker="o", linewidth=1.2, color=col, linestyle=ls)
                    y_for_ylim.extend([float(v) for v in y if np.isfinite(v)])
                    if "p_sem" in ss.columns:
                        se = pd.to_numeric(ss["p_sem"], errors="coerce").to_numpy(dtype=float)
                        lo = np.clip(y - se, 0.0, 1.0)
                        hi = np.clip(y + se, 0.0, 1.0)
                        ax.fill_between(x, lo, hi, color=col, alpha=0.12)
                        y_for_ylim.extend([float(v) for v in hi if np.isfinite(v)])

            ax.set_title(f"{choice_label}: {subset_label}")
            ax.set_xlabel("Fixation position")
            ax.set_ylabel("P(sign & within-subset |value|-rank bin | pos)")
            _set_prob_ylim(ax, y_for_ylim)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(x_vals)

            if row_i == 0 and col_j == 0:
                from matplotlib.lines import Line2D

                bin_handles = [
                    Line2D([0], [0], color=bin_colors[b], lw=2, label=f"bin {b}")
                    for b in [1, 2, 3]
                ]
                sign_handles = [
                    Line2D([0], [0], color="black", lw=2, linestyle=sign_styles["pos"], label="positive"),
                    Line2D([0], [0], color="black", lw=2, linestyle=sign_styles["neg"], label="negative"),
                ]
                ax.legend(handles=sign_handles + bin_handles, frameon=False, fontsize=8, ncol=2)

    fig.suptitle(
        "Early-fixation targeting by sign + within-subset magnitude (P(sign & within-subset |value|-rank bin | position))"
    )
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_summary(
    summary: pd.DataFrame,
    rank6_long: pd.DataFrame,
    rel3_long: pd.DataFrame,
    irrel3_long: pd.DataFrame,
    *,
    out_png: Path,
    out_pdf: Path,
) -> None:
    order = ["overall", "take", "leave"]

    def _rank_bin_label(bin_idx: int, *, n_bins: int, rank_mode: Optional[str]) -> str:
        if rank_mode == "signed":
            low = "lowest value"
            high = "highest value"
        else:
            # Default to abs-mode semantics (overall row).
            low = "lowest |value|"
            high = "highest |value|"

        if bin_idx == 1:
            return f"rank 1 ({low})"
        if bin_idx == n_bins:
            return f"rank {n_bins} ({high})"
        return f"rank {bin_idx}"

    def _set_prob_ylim(ax: plt.Axes, y_values: Iterable[float]) -> None:
        vals = [float(v) for v in y_values if v is not None and np.isfinite(v)]
        if not vals:
            ax.set_ylim(0.0, 1.0)
            return
        ymax = max(vals)
        upper = min(1.0, max(0.05, ymax * 1.10, ymax + 0.02))
        ax.set_ylim(0.0, upper)

    fig, axes = plt.subplots(3, 4, figsize=(18, 10.5), constrained_layout=True)

    for row_i, label in enumerate(order):
        s = summary[summary.get("choice_split") == label].copy()
        r6 = rank6_long[rank6_long.get("choice_split") == label].copy()
        r3 = rel3_long[rel3_long.get("choice_split") == label].copy()
        ir3 = irrel3_long[irrel3_long.get("choice_split") == label].copy()

        row_rank_mode = None
        if not s.empty and "rank_mode" in s.columns:
            u = sorted(s["rank_mode"].dropna().unique().tolist())
            row_rank_mode = u[0] if len(u) == 1 else None

        ax0 = axes[row_i, 0]
        if s.empty:
            ax0.set_title(f"{label}: (no trials)")
            ax0.axis("off")
        else:
            x = s["fixation_position"].to_numpy(dtype=int)

            def _plot_mean_sem(y_col: str, label_text: str) -> None:
                y = pd.to_numeric(s.get(y_col), errors="coerce").to_numpy(dtype=float)
                ax0.plot(x, y, marker="o", label=label_text)
                sem_col = f"{y_col}_sem"
                if sem_col in s.columns:
                    se = pd.to_numeric(s.get(sem_col), errors="coerce").to_numpy(dtype=float)
                    lo = np.clip(y - se, 0.0, 1.0)
                    hi = np.clip(y + se, 0.0, 1.0)
                    ax0.fill_between(x, lo, hi, alpha=0.20)

            _plot_mean_sem("p_relevant", "P(relevant)")
            ax0.set_title(label)
            ax0.set_xlabel("Fixation position")
            ax0.set_ylabel("Probability")
            y_cols = [
                "p_relevant",
                "p_relevant_sem",
            ]
            y_series = [pd.to_numeric(s[c], errors="coerce") for c in y_cols if c in s.columns]
            y_for_ylim = pd.concat(y_series).dropna().to_numpy(dtype=float) if y_series else np.array([])
            _set_prob_ylim(ax0, y_for_ylim)
            ax0.grid(True, alpha=0.3)
            if row_i == 0:
                ax0.legend(frameon=False, fontsize=8)

        ax1 = axes[row_i, 1]
        if r6.empty:
            ax1.set_title("rank (6): (no data)")
            ax1.axis("off")
        else:
            pos = sorted(pd.to_numeric(r6["fixation_position"], errors="coerce").dropna().astype(int).unique().tolist())
            piv = r6.pivot_table(index="fixation_position", columns="rank_bin", values="p", aggfunc="mean").reindex(pos)
            piv_sem = None
            if "p_sem" in r6.columns:
                piv_sem = r6.pivot_table(index="fixation_position", columns="rank_bin", values="p_sem", aggfunc="mean").reindex(pos)
            for b in range(1, 7):
                if b not in piv.columns:
                    continue
                y = piv[b].to_numpy(dtype=float)
                ax1.plot(pos, y, marker="o", linewidth=1.5, label=_rank_bin_label(b, n_bins=6, rank_mode=row_rank_mode))
                if piv_sem is not None and b in piv_sem.columns:
                    se = piv_sem[b].to_numpy(dtype=float)
                    lo = np.clip(y - se, 0.0, 1.0)
                    hi = np.clip(y + se, 0.0, 1.0)
                    ax1.fill_between(pos, lo, hi, alpha=0.15)
            ax1.set_title("recalled value rank (6 items)" if row_rank_mode == "signed" else "|recalled| rank (6 items)")
            ax1.set_xlabel("Fixation position")
            ax1.set_ylabel("P(bin | position)")
            ax1.grid(True, alpha=0.3)
            y_for_ylim = r6["p"].to_numpy(dtype=float)
            if "p_sem" in r6.columns:
                y_for_ylim = np.concatenate([y_for_ylim, np.clip((r6["p"] + r6["p_sem"]).to_numpy(dtype=float), 0.0, 1.0)])
            _set_prob_ylim(ax1, y_for_ylim)
            if row_i == 0:
                ax1.legend(frameon=False, fontsize=7, ncol=2)

        ax2 = axes[row_i, 2]
        if r3.empty:
            ax2.set_title("rank (3 relevant): (no data)")
            ax2.axis("off")
        else:
            pos3 = sorted(pd.to_numeric(r3["fixation_position"], errors="coerce").dropna().astype(int).unique().tolist())
            piv3 = r3.pivot_table(index="fixation_position", columns="rank_bin", values="p", aggfunc="mean").reindex(pos3)
            piv3_sem = None
            if "p_sem" in r3.columns:
                piv3_sem = r3.pivot_table(index="fixation_position", columns="rank_bin", values="p_sem", aggfunc="mean").reindex(pos3)
            for b in range(1, 4):
                if b not in piv3.columns:
                    continue
                y = piv3[b].to_numpy(dtype=float)
                ax2.plot(pos3, y, marker="o", linewidth=1.5, label=_rank_bin_label(b, n_bins=3, rank_mode=row_rank_mode))
                if piv3_sem is not None and b in piv3_sem.columns:
                    se = piv3_sem[b].to_numpy(dtype=float)
                    lo = np.clip(y - se, 0.0, 1.0)
                    hi = np.clip(y + se, 0.0, 1.0)
                    ax2.fill_between(pos3, lo, hi, alpha=0.15)
            ax2.set_title(
                "recalled value rank (3 relevant)\nP(bin | relevant, pos)"
                if row_rank_mode == "signed"
                else "|recalled| rank (3 relevant)\nP(bin | relevant, pos)"
            )
            ax2.set_xlabel("Fixation position")
            ax2.set_ylabel("P(bin | relevant, pos)")
            ax2.grid(True, alpha=0.3)
            y_for_ylim = r3["p"].to_numpy(dtype=float)
            if "p_sem" in r3.columns:
                y_for_ylim = np.concatenate([y_for_ylim, np.clip((r3["p"] + r3["p_sem"]).to_numpy(dtype=float), 0.0, 1.0)])
            _set_prob_ylim(ax2, y_for_ylim)
            if row_i == 0:
                ax2.legend(frameon=False, fontsize=7, ncol=1)

        # Column 4: irrelevant-only rank among 3 distribution (conditional)
        ax3 = axes[row_i, 3]
        if ir3.empty:
            ax3.set_title("rank (3 irrelevant): (no data)")
            ax3.axis("off")
        else:
            posi = sorted(pd.to_numeric(ir3["fixation_position"], errors="coerce").dropna().astype(int).unique().tolist())
            pivi = ir3.pivot_table(index="fixation_position", columns="rank_bin", values="p", aggfunc="mean").reindex(posi)
            pivi_sem = None
            if "p_sem" in ir3.columns:
                pivi_sem = ir3.pivot_table(index="fixation_position", columns="rank_bin", values="p_sem", aggfunc="mean").reindex(posi)
            for b in range(1, 4):
                if b not in pivi.columns:
                    continue
                y = pivi[b].to_numpy(dtype=float)
                ax3.plot(posi, y, marker="o", linewidth=1.5, label=_rank_bin_label(b, n_bins=3, rank_mode=row_rank_mode))
                if pivi_sem is not None and b in pivi_sem.columns:
                    se = pivi_sem[b].to_numpy(dtype=float)
                    lo = np.clip(y - se, 0.0, 1.0)
                    hi = np.clip(y + se, 0.0, 1.0)
                    ax3.fill_between(posi, lo, hi, alpha=0.15)
            ax3.set_title(
                "recalled value rank (3 irrelevant)\nP(bin | irrelevant, pos)"
                if row_rank_mode == "signed"
                else "|recalled| rank (3 irrelevant)\nP(bin | irrelevant, pos)"
            )
            ax3.set_xlabel("Fixation position")
            ax3.set_ylabel("P(bin | irrelevant, pos)")
            ax3.grid(True, alpha=0.3)
            y_for_ylim = ir3["p"].to_numpy(dtype=float)
            if "p_sem" in ir3.columns:
                y_for_ylim = np.concatenate([y_for_ylim, np.clip((ir3["p"] + ir3["p_sem"]).to_numpy(dtype=float), 0.0, 1.0)])
            _set_prob_ylim(ax3, y_for_ylim)
            if row_i == 0:
                ax3.legend(frameon=False, fontsize=7, ncol=1)

    fig.suptitle("Early-fixation targeting by choice (overall / take / leave)")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize early choice fixations (humans): relevance and recalled-value targeting. "
            "Uses recalled reward rank bins per game (6 items) and within relevant items (3 items)."
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
    parser.add_argument(
        "--split-by-trial-metrics",
        action="store_true",
        help=(
            "Also generate DO vs DONT versions of the plots/CSVs for each trial-level metric flag "
            "(`is_*_trial`) from analysis/analyze_critical_items.py outputs."
        ),
    )
    parser.add_argument(
        "--trial-metrics-csv",
        type=str,
        default="",
        help=(
            "Path to a trial-metrics CSV such as figures/critical_trials_human_true*.csv. "
            "If omitted and --split-by-trial-metrics is set, auto-detect under <base-dir>/figures/."
        ),
    )
    parser.add_argument(
        "--reward-source",
        type=str,
        default="recalled",
        choices=["recalled", "true"],
        help=(
            "Which item-value source to use for ranking. 'recalled' parses valuerecall transcripts; "
            "'true' uses encoded rewards from the main logfile (useful for NN simulations)."
        ),
    )
    parser.add_argument(
        "--missing-recall-fallback",
        type=str,
        default="true",
        choices=["true", "lowest"],
        help=(
            "How to handle missing/unparseable recalled values in valuerecall transcripts. "
            "'true' fills NaN recalled values with the true encoded reward; "
            "'lowest' leaves NaN in place so missing is treated as lowest rank bin."
        ),
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

    recalled = _load_reward_table_for_fixations(
        base_dir,
        subjects=sorted(fix["subject_id"].unique().tolist()),
        excluded_subjects=tuple(args.exclude_subjects),
        reward_source=str(args.reward_source),
        missing_recall_fallback=str(args.missing_recall_fallback),
    )

    if recalled is None or recalled.empty:
        raise RuntimeError(
            "No item-value table could be built for ranking. "
            "For NN simulations, try --reward-source true."
        )

    out_dir = base_dir / "output"
    fig_dir = base_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tag = str(args.tag).strip()
    tag_sfx = f"_{tag}" if tag else ""

    def _run_and_save(fix_subset: pd.DataFrame, *, name_prefix: str) -> None:
        if fix_subset.empty:
            print(f"Skipping {name_prefix}: no fixations")
            return

        summary, rank6_long, rel3_long, irrel3_long = compute_all_summaries(
            fix_subset, recalled, max_fixations=int(args.max_fixations)
        )

        # Separate sign-based summary (positive vs negative recalled values)
        sign_frames: List[pd.DataFrame] = []
        for label, sub in [
            ("overall", fix_subset),
            ("take", fix_subset[fix_subset["choice"] == 1].copy()),
            ("leave", fix_subset[fix_subset["choice"] == 2].copy()),
        ]:
            if sub.empty:
                continue
            for subset in ["overall", "relevant", "irrelevant"]:
                ss = build_sign_summary(sub, recalled, max_fixations=int(args.max_fixations), subset=subset)
                ss["choice_split"] = label
                sign_frames.append(ss)
        sign_summary = pd.concat(sign_frames, ignore_index=True) if sign_frames else pd.DataFrame()

        # Sign + magnitude (abs-rank6) summary
        sign_rank_frames: List[pd.DataFrame] = []
        for label, sub in [
            ("overall", fix_subset),
            ("take", fix_subset[fix_subset["choice"] == 1].copy()),
            ("leave", fix_subset[fix_subset["choice"] == 2].copy()),
        ]:
            if sub.empty:
                continue
            for subset in ["overall", "relevant", "irrelevant"]:
                sr = build_absrank6_by_sign_summary(sub, recalled, max_fixations=int(args.max_fixations), subset=subset)
                if sr is None or sr.empty:
                    continue
                sr["choice_split"] = label
                sign_rank_frames.append(sr)
        sign_rank_summary = pd.concat(sign_rank_frames, ignore_index=True) if sign_rank_frames else pd.DataFrame()

        # Joint distribution over (sign x abs-rank6 bin), out of all fixations at each position.
        sign_rankdist_frames: List[pd.DataFrame] = []
        for label, sub in [
            ("overall", fix_subset),
            ("take", fix_subset[fix_subset["choice"] == 1].copy()),
            ("leave", fix_subset[fix_subset["choice"] == 2].copy()),
        ]:
            if sub.empty:
                continue
            for subset in ["overall", "relevant", "irrelevant"]:
                srd = build_absrank6_dist_by_sign_joint(sub, recalled, max_fixations=int(args.max_fixations), subset=subset)
                if srd is None or srd.empty:
                    continue
                srd["choice_split"] = label
                sign_rankdist_frames.append(srd)
        sign_rankdist = pd.concat(sign_rankdist_frames, ignore_index=True) if sign_rankdist_frames else pd.DataFrame()

        # Joint distribution over (sign x within-subset abs-rank3 bin) for relevant/irrelevant.
        within3_frames: List[pd.DataFrame] = []
        for label, sub in [
            ("overall", fix_subset),
            ("take", fix_subset[fix_subset["choice"] == 1].copy()),
            ("leave", fix_subset[fix_subset["choice"] == 2].copy()),
        ]:
            if sub.empty:
                continue
            for subset in ["relevant", "irrelevant"]:
                w3 = build_absrank3_within_subset_dist_by_sign_joint(
                    sub,
                    recalled,
                    max_fixations=int(args.max_fixations),
                    subset=subset,
                )
                if w3 is None or w3.empty:
                    continue
                w3["choice_split"] = label
                within3_frames.append(w3)
        within3_rankdist = pd.concat(within3_frames, ignore_index=True) if within3_frames else pd.DataFrame()

        prefix_sfx = f"_{name_prefix}" if str(name_prefix).strip() else ""

        if not prefix_sfx:
            # Default filenames for the all-trials run.
            out_csv = out_dir / f"first_fixations_relevance_magnitude_max{int(args.max_fixations)}{tag_sfx}.csv"
            out_rank6_csv = out_dir / f"first_fixations_absrank6_dist_max{int(args.max_fixations)}{tag_sfx}.csv"
            out_rel3_csv = out_dir / f"first_fixations_absrankrel3_dist_max{int(args.max_fixations)}{tag_sfx}.csv"
            out_irrel3_csv = out_dir / f"first_fixations_absrankirrel3_dist_max{int(args.max_fixations)}{tag_sfx}.csv"
            out_png = fig_dir / f"first_fixations_relevance_magnitude_max{int(args.max_fixations)}{tag_sfx}.png"
            out_pdf = fig_dir / f"first_fixations_relevance_magnitude_max{int(args.max_fixations)}{tag_sfx}.pdf"

            out_sign_csv = out_dir / f"first_fixations_recalled_sign_max{int(args.max_fixations)}{tag_sfx}.csv"
            out_sign_png = fig_dir / f"first_fixations_recalled_sign_max{int(args.max_fixations)}{tag_sfx}.png"
            out_sign_pdf = fig_dir / f"first_fixations_recalled_sign_max{int(args.max_fixations)}{tag_sfx}.pdf"

            out_signrank_csv = out_dir / f"first_fixations_absrank6_by_sign_max{int(args.max_fixations)}{tag_sfx}.csv"
            out_signrank_png = fig_dir / f"first_fixations_absrank6_by_sign_max{int(args.max_fixations)}{tag_sfx}.png"
            out_signrank_pdf = fig_dir / f"first_fixations_absrank6_by_sign_max{int(args.max_fixations)}{tag_sfx}.pdf"

            out_signrankdist_csv = out_dir / f"first_fixations_absrank6_by_sign_dist_max{int(args.max_fixations)}{tag_sfx}.csv"
            out_signrankdist_png = fig_dir / f"first_fixations_absrank6_by_sign_dist_max{int(args.max_fixations)}{tag_sfx}.png"
            out_signrankdist_pdf = fig_dir / f"first_fixations_absrank6_by_sign_dist_max{int(args.max_fixations)}{tag_sfx}.pdf"

            out_within3_csv = out_dir / f"first_fixations_absrank3_within_subset_by_sign_dist_max{int(args.max_fixations)}{tag_sfx}.csv"
            out_within3_png = fig_dir / f"first_fixations_absrank3_within_subset_by_sign_dist_max{int(args.max_fixations)}{tag_sfx}.png"
            out_within3_pdf = fig_dir / f"first_fixations_absrank3_within_subset_by_sign_dist_max{int(args.max_fixations)}{tag_sfx}.pdf"
        else:
            out_csv = out_dir / f"first_fixations_relevance_magnitude{prefix_sfx}_max{int(args.max_fixations)}{tag_sfx}.csv"
            out_rank6_csv = out_dir / f"first_fixations_absrank6_dist{prefix_sfx}_max{int(args.max_fixations)}{tag_sfx}.csv"
            out_rel3_csv = out_dir / f"first_fixations_absrankrel3_dist{prefix_sfx}_max{int(args.max_fixations)}{tag_sfx}.csv"
            out_irrel3_csv = out_dir / f"first_fixations_absrankirrel3_dist{prefix_sfx}_max{int(args.max_fixations)}{tag_sfx}.csv"
            out_png = fig_dir / f"first_fixations_relevance_magnitude{prefix_sfx}_max{int(args.max_fixations)}{tag_sfx}.png"
            out_pdf = fig_dir / f"first_fixations_relevance_magnitude{prefix_sfx}_max{int(args.max_fixations)}{tag_sfx}.pdf"

            out_sign_csv = out_dir / f"first_fixations_recalled_sign{prefix_sfx}_max{int(args.max_fixations)}{tag_sfx}.csv"
            out_sign_png = fig_dir / f"first_fixations_recalled_sign{prefix_sfx}_max{int(args.max_fixations)}{tag_sfx}.png"
            out_sign_pdf = fig_dir / f"first_fixations_recalled_sign{prefix_sfx}_max{int(args.max_fixations)}{tag_sfx}.pdf"

            out_signrank_csv = out_dir / f"first_fixations_absrank6_by_sign{prefix_sfx}_max{int(args.max_fixations)}{tag_sfx}.csv"
            out_signrank_png = fig_dir / f"first_fixations_absrank6_by_sign{prefix_sfx}_max{int(args.max_fixations)}{tag_sfx}.png"
            out_signrank_pdf = fig_dir / f"first_fixations_absrank6_by_sign{prefix_sfx}_max{int(args.max_fixations)}{tag_sfx}.pdf"

            out_signrankdist_csv = out_dir / f"first_fixations_absrank6_by_sign_dist{prefix_sfx}_max{int(args.max_fixations)}{tag_sfx}.csv"
            out_signrankdist_png = fig_dir / f"first_fixations_absrank6_by_sign_dist{prefix_sfx}_max{int(args.max_fixations)}{tag_sfx}.png"
            out_signrankdist_pdf = fig_dir / f"first_fixations_absrank6_by_sign_dist{prefix_sfx}_max{int(args.max_fixations)}{tag_sfx}.pdf"

            out_within3_csv = out_dir / f"first_fixations_absrank3_within_subset_by_sign_dist{prefix_sfx}_max{int(args.max_fixations)}{tag_sfx}.csv"
            out_within3_png = fig_dir / f"first_fixations_absrank3_within_subset_by_sign_dist{prefix_sfx}_max{int(args.max_fixations)}{tag_sfx}.png"
            out_within3_pdf = fig_dir / f"first_fixations_absrank3_within_subset_by_sign_dist{prefix_sfx}_max{int(args.max_fixations)}{tag_sfx}.pdf"

        summary.to_csv(out_csv, index=False)
        rank6_long.to_csv(out_rank6_csv, index=False)
        rel3_long.to_csv(out_rel3_csv, index=False)
        irrel3_long.to_csv(out_irrel3_csv, index=False)
        plot_summary(summary, rank6_long, rel3_long, irrel3_long, out_png=out_png, out_pdf=out_pdf)

        if not sign_summary.empty:
            sign_summary.to_csv(out_sign_csv, index=False)
            plot_sign_summary(sign_summary, out_png=out_sign_png, out_pdf=out_sign_pdf)

        if not sign_rank_summary.empty:
            sign_rank_summary.to_csv(out_signrank_csv, index=False)
            plot_absrank6_by_sign_summary(sign_rank_summary, out_png=out_signrank_png, out_pdf=out_signrank_pdf)

        if not sign_rankdist.empty:
            sign_rankdist.to_csv(out_signrankdist_csv, index=False)
            plot_absrank6_dist_by_sign_joint(sign_rankdist, out_png=out_signrankdist_png, out_pdf=out_signrankdist_pdf)

        if not within3_rankdist.empty:
            within3_rankdist.to_csv(out_within3_csv, index=False)
            plot_absrank3_within_subset_dist_by_sign_joint(
                within3_rankdist,
                out_png=out_within3_png,
                out_pdf=out_within3_pdf,
            )

        print(f"Saved summary CSV: {out_csv}")
        print(f"Saved figure: {out_png}")
        if not sign_summary.empty:
            print(f"Saved sign summary CSV: {out_sign_csv}")
            print(f"Saved sign figure: {out_sign_png}")
        if not sign_rank_summary.empty:
            print(f"Saved sign+rank summary CSV: {out_signrank_csv}")
            print(f"Saved sign+rank figure: {out_signrank_png}")
        if not sign_rankdist.empty:
            print(f"Saved sign+rank dist CSV: {out_signrankdist_csv}")
            print(f"Saved sign+rank dist figure: {out_signrankdist_png}")
        if not within3_rankdist.empty:
            print(f"Saved within-subset sign+rank dist CSV: {out_within3_csv}")
            print(f"Saved within-subset sign+rank dist figure: {out_within3_png}")

    # All trials
    _run_and_save(fix, name_prefix="")

    # Optional: DO vs DONT splits per trial-level metric.
    if bool(args.split_by_trial_metrics):
        metrics_path: Optional[Path]
        if str(args.trial_metrics_csv).strip():
            metrics_path = Path(str(args.trial_metrics_csv)).expanduser().resolve()
        else:
            metrics_path = _find_default_trial_metrics_csv(base_dir)

        if metrics_path is None or not metrics_path.exists():
            raise FileNotFoundError(
                "--split-by-trial-metrics was set but no trial metrics CSV was found. "
                "Provide --trial-metrics-csv or run analysis/analyze_critical_items.py to generate figures/critical_trials_human_true*.csv"
            )

        metrics = load_trial_metrics(metrics_path)
        metric_cols = _infer_trial_metric_flag_columns(metrics)
        if not metric_cols:
            raise RuntimeError(f"No binary is_*_trial columns found in {metrics_path}")

        print(f"Loaded trial metrics: {metrics_path}")
        print(f"Splitting by metrics: {metric_cols}")

        for mc in metric_cols:
            do_fix = _subset_fixations_by_trial_metric(fix, metrics, metric_col=mc, desired_value=1)
            dont_fix = _subset_fixations_by_trial_metric(fix, metrics, metric_col=mc, desired_value=0)
            _run_and_save(do_fix, name_prefix=f"{mc}_do")
            _run_and_save(dont_fix, name_prefix=f"{mc}_dont")

    print(f"Loaded clean fixations: {in_path}")
    print(
        f"Subjects: {fix['subject_id'].nunique()} | "
        f"Trials: {fix[['subject_id','game','trial_number']].drop_duplicates().shape[0]} | "
        f"Fixations: {len(fix)}"
    )


if __name__ == "__main__":
    main()
