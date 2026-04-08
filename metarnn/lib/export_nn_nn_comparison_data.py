#!/usr/bin/env python3
"""Export subject-level data for brms human-vs-NN statistical comparisons.

Produces CSVs consumed by run_mixed_effects_nn_nn_comparison.R, reusing
compute functions from plot_NN_H_comparison.py for consistency.

Example:
  conda run -n analysis python metarnn/lib/export_nn_nn_comparison_data.py \\
    --nn-root metarnn/simulations/human_like_04_04_input5 \\
    --tag 04_04_input5 \\
    --out-dir metarnn/simulations/human_like_04_04_input5/output/human_vs_nn_brms/data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.lib.analyze_fixation_duration_by_position import (  # noqa: E402
    COLLAPSE_KEEP_FIRST_N,
    DEFAULT_EXCLUDED as DEFAULT_EXCLUDED_EYE,
    load_clean_choice_fixations as load_clean_choice_fixations_full,
)
from metarnn.lib.plot_NN_H_comparison import (  # noqa: E402
    _find_default_clean_choice_fixations,
    _find_revisits_by_subject_csv,
    _load_clean_choice_fixations,
    _subject_cumtime_curve,
)
from analysis.lib.visualize_first_fixations_relevance_and_magnitude import (  # noqa: E402
    load_recalled_rewards,
)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Panel A: revisit bars — subject-level firstfix / revisit proportions
# ---------------------------------------------------------------------------


def export_panel_A(
    human_stats_dir: Path,
    nn_stats_dir: Path,
    tag: str,
    out_dir: Path,
) -> Path:
    human_csv = _find_revisits_by_subject_csv(
        human_stats_dir, agent_tag="human",
    )
    nn_csv = _find_revisits_by_subject_csv(
        nn_stats_dir, agent_tag="nn", tag=tag,
    )

    cols = ["subject_id", "firstfix_prop_time_relevant", "revisit_prop_time_relevant"]

    h = pd.read_csv(human_csv)
    # Column may be named 'subject' or 'subject_id' depending on upstream script
    if "subject" in h.columns and "subject_id" not in h.columns:
        h = h.rename(columns={"subject": "subject_id"})
    h["subject_id"] = "H" + h["subject_id"].astype(str)
    h["group"] = "human"

    n = pd.read_csv(nn_csv)
    if "subject" in n.columns and "subject_id" not in n.columns:
        n = n.rename(columns={"subject": "subject_id"})
    n["subject_id"] = "N" + n["subject_id"].astype(str)
    n["group"] = "nn"

    out = pd.concat([h[cols + ["group"]], n[cols + ["group"]]], ignore_index=True)
    out_path = out_dir / f"panelA_subject_data_{tag}.csv"
    out.to_csv(out_path, index=False)
    print(f"  Panel A: {out_path}  ({len(out)} rows)")
    return out_path


# ---------------------------------------------------------------------------
# Panel B: cumulative fixation-time AUC and 80 % crossing
# ---------------------------------------------------------------------------


def export_panel_B(
    nn_root: Path,
    tag: str,
    out_dir: Path,
    *,
    max_fixations: int = 40,
    cumtime_mode: str = "all_trials",
    excluded_subjects: tuple[str, ...] = tuple(DEFAULT_EXCLUDED_EYE),
) -> Path:
    rows: list[dict] = []

    for group_label, base_dir in [("human", _REPO_ROOT), ("nn", nn_root)]:
        clean_path = _find_default_clean_choice_fixations(base_dir)
        fix = _load_clean_choice_fixations(clean_path)

        excluded = set(map(str, excluded_subjects))
        if group_label == "human" and excluded:
            fix = fix[~fix["subject_id"].astype(str).isin(excluded)].copy()

        for sid, dsub in fix.groupby("subject_id", sort=True):
            try:
                curve = _subject_cumtime_curve(
                    dsub, max_fixations=max_fixations, mode=cumtime_mode,
                )
            except Exception:
                continue
            if np.all(~np.isfinite(curve)):
                continue

            # AUC normalised to [0, 1]
            _trapz = getattr(np, "trapezoid", np.trapz)  # numpy >=2.0 compat
            auc = float(_trapz(curve, dx=1) / (max_fixations - 1))

            # First fixation position where cumulative proportion >= 0.8
            hits = np.where(curve >= 0.8)[0]
            crossing = int(hits[0] + 1) if len(hits) > 0 else np.nan

            prefix = "H" if group_label == "human" else "N"
            rows.append({
                "subject_id": f"{prefix}{sid}",
                "group": group_label,
                "auc": auc,
                "crossing_80": crossing,
            })

    out = pd.DataFrame(rows)
    out_path = out_dir / f"panelB_cumtime_subject_data_{tag}.csv"
    out.to_csv(out_path, index=False)
    print(f"  Panel B: {out_path}  ({len(out)} rows)")
    return out_path


# ---------------------------------------------------------------------------
# Panel E/F: proportion relevant fixated by position (subject-level)
# ---------------------------------------------------------------------------


def _compute_prop_relevant_per_subject(
    fix: pd.DataFrame,
    *,
    max_fixations: int = 8,
    keep_first_n: int = COLLAPSE_KEEP_FIRST_N,
) -> pd.DataFrame:
    """Return subject × position means of P(relevant | fixation position).

    Same logic as _compute_prop_relevant_by_position_7plus but returns per_sub
    instead of group summary.
    """
    if fix is None or fix.empty:
        return pd.DataFrame()

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
        tail_agg = tail.groupby(trial_cols, as_index=False).agg(
            prop_relevant=("is_relevant", "mean"),
        )
        tail_agg["fixation_position"] = int(keep_first_n) + 1
        tail_agg = tail_agg[trial_cols + ["fixation_position", "prop_relevant"]]
        collapsed = pd.concat([early, tail_agg], ignore_index=True)
    else:
        collapsed = early

    if collapsed.empty:
        return pd.DataFrame()

    per_sub = (
        collapsed.groupby(["subject_id", "fixation_position"], as_index=False)
        .agg(prop_relevant=("prop_relevant", "mean"))
        .copy()
    )
    return per_sub


def export_panel_EF(
    nn_root: Path,
    tag: str,
    out_dir: Path,
    *,
    max_fixations: int = 8,
    keep_first_n: int = COLLAPSE_KEEP_FIRST_N,
    excluded_subjects: tuple[str, ...] = tuple(DEFAULT_EXCLUDED_EYE),
    filename_suffix: str = "",
) -> Path:
    frames: list[pd.DataFrame] = []

    for group_label, base_dir in [("human", _REPO_ROOT), ("nn", nn_root)]:
        clean_path = _find_default_clean_choice_fixations(base_dir)
        fix = load_clean_choice_fixations_full(clean_path)

        excluded = set(map(str, excluded_subjects))
        if group_label == "human" and excluded:
            fix = fix[~fix["subject_id"].astype(str).isin(excluded)].copy()

        per_sub = _compute_prop_relevant_per_subject(
            fix, max_fixations=max_fixations, keep_first_n=keep_first_n,
        )
        if per_sub.empty:
            continue

        prefix = "H" if group_label == "human" else "N"
        per_sub["subject_id"] = prefix + per_sub["subject_id"].astype(str)
        per_sub["group"] = group_label
        frames.append(per_sub)

    out = pd.concat(frames, ignore_index=True)
    out_path = out_dir / f"panelEF_prop_relevant_subject_data_{tag}{filename_suffix}.csv"
    out.to_csv(out_path, index=False)
    print(f"  Panel E/F{filename_suffix}: {out_path}  ({len(out)} rows)")
    return out_path


# ---------------------------------------------------------------------------
# Panel G/H: valence diff by position for take / leave (subject-level)
# ---------------------------------------------------------------------------


def _compute_valence_diff_per_subject(
    fix: pd.DataFrame,
    *,
    decision: int,
    max_fixations: int = 8,
    keep_first_n: int = COLLAPSE_KEEP_FIRST_N,
    denom: str = "all",
    reward_col: str = "reward",
) -> pd.DataFrame:
    """Return subject × position valence_diff = prop_positive - prop_negative.

    Same logic as _compute_prop_positive_negative_relevant_by_decision
    but returns per_sub level pivoted to a diff column.
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

    # Denominator
    if denom == "all":
        d_total = d.copy()
    else:
        d_total = d[d["is_relevant"] == 1].copy()
        d_total[reward_col] = pd.to_numeric(d_total[reward_col], errors="coerce")
        d_total = d_total.dropna(subset=[reward_col])
        d_total = d_total[d_total[reward_col] != 0]

    # Numerator: relevant items with known non-zero reward
    d_num = d[d["is_relevant"] == 1].copy()
    d_num[reward_col] = pd.to_numeric(d_num[reward_col], errors="coerce")
    d_num = d_num.dropna(subset=[reward_col])
    d_num = d_num[d_num[reward_col] != 0]

    if d_total.empty or d_num.empty:
        return pd.DataFrame()

    d_num["valence"] = np.where(d_num[reward_col] > 0, "positive", "negative")

    counts = d_num.groupby(
        trial_cols + ["pos_collapsed", "valence"], as_index=False,
    ).agg(n_fix=("valence", "size"))

    totals = d_total.groupby(
        trial_cols + ["pos_collapsed"], as_index=False,
    ).agg(n_total=("pos_collapsed", "size"))
    totals = totals[totals["n_total"] > 0].copy()

    vals = pd.DataFrame({"valence": ["positive", "negative"]})
    grid = totals.assign(_k=1).merge(vals.assign(_k=1), on="_k").drop(columns=["_k"])
    counts = grid.merge(counts, on=trial_cols + ["pos_collapsed", "valence"], how="left")
    counts["n_fix"] = pd.to_numeric(counts["n_fix"], errors="coerce").fillna(0.0)
    counts["prop"] = counts["n_fix"] / counts["n_total"]

    # Within-subject means
    per_sub = counts.groupby(
        ["subject_id", "pos_collapsed", "valence"], as_index=False,
    ).agg(prop=("prop", "mean"))

    # Pivot prop
    pivoted = per_sub.pivot_table(
        index=["subject_id", "pos_collapsed"],
        columns="valence",
        values="prop",
        aggfunc="mean",
    ).reset_index()

    if "positive" not in pivoted.columns or "negative" not in pivoted.columns:
        return pd.DataFrame()

    pivoted["valence_diff"] = pivoted["positive"] - pivoted["negative"]

    result = pivoted[[
        "subject_id", "pos_collapsed", "valence_diff",
        "positive", "negative",
    ]].copy()
    result = result.rename(columns={
        "pos_collapsed": "fixation_position",
        "positive": "prop_positive",
        "negative": "prop_negative",
    })
    return result


def export_panel_GH(
    nn_root: Path,
    tag: str,
    out_dir: Path,
    *,
    max_fixations: int = 8,
    keep_first_n: int = COLLAPSE_KEEP_FIRST_N,
    excluded_subjects: tuple[str, ...] = tuple(DEFAULT_EXCLUDED_EYE),
    denom: str = "all",
    filename_suffix: str = "",
) -> Path:
    frames: list[pd.DataFrame] = []

    for group_label, base_dir in [("human", _REPO_ROOT), ("nn", nn_root)]:
        clean_path = _find_default_clean_choice_fixations(base_dir)
        fix = load_clean_choice_fixations_full(clean_path)

        excluded = set(map(str, excluded_subjects))
        if group_label == "human" and excluded:
            fix = fix[~fix["subject_id"].astype(str).isin(excluded)].copy()

        # Human uses recalled rewards; NN uses true rewards
        reward_col = "reward"
        if group_label == "human" and "reward_recalled" not in fix.columns:
            recalled = load_recalled_rewards(
                base_dir, excluded_subjects=excluded_subjects,
            )
            fix = fix.merge(
                recalled[["subject_id", "game", "image", "reward_recalled"]],
                on=["subject_id", "game", "image"],
                how="left",
            )
            reward_col = "reward_recalled"
        elif group_label == "human":
            reward_col = "reward_recalled"

        for decision, dec_label in [(1, "take"), (2, "leave")]:
            per_sub = _compute_valence_diff_per_subject(
                fix,
                decision=decision,
                max_fixations=max_fixations,
                keep_first_n=keep_first_n,
                denom=denom,
                reward_col=reward_col,
            )
            if per_sub.empty:
                continue

            prefix = "H" if group_label == "human" else "N"
            per_sub["subject_id"] = prefix + per_sub["subject_id"].astype(str)
            per_sub["group"] = group_label
            per_sub["decision"] = dec_label
            frames.append(per_sub)

    out = pd.concat(frames, ignore_index=True)
    out_path = out_dir / f"panelGH_valence_diff_subject_data_{tag}{filename_suffix}.csv"
    out.to_csv(out_path, index=False)
    print(f"  Panel G/H{filename_suffix}: {out_path}  ({len(out)} rows)")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export subject-level data for brms human-vs-NN comparisons.",
    )
    parser.add_argument("--nn-root", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--max-fixations", type=int, default=40,
                        help="Max fixation for cumtime curve (Panel B).")
    parser.add_argument("--cumtime-mode", type=str, default="all_trials",
                        choices=["all_trials", "conditional"])
    parser.add_argument("--gh-denom", type=str, default="all",
                        choices=["relevant", "all"],
                        help="Denominator for Panel G/H valence proportions.")

    args = parser.parse_args()

    nn_root = Path(args.nn_root).resolve()

    human_stats_dir = _REPO_ROOT / "output" / "eyegaze" / "stats"
    nn_stats_dir = nn_root / "output" / "eyegaze" / "stats"

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = nn_root / "output" / "human_vs_nn_brms" / "data"

    _ensure_dir(out_dir)
    print(f"Exporting data for tag={args.tag}, nn_root={nn_root}")
    print(f"Output: {out_dir}\n")

    export_panel_A(
        human_stats_dir=human_stats_dir,
        nn_stats_dir=nn_stats_dir,
        tag=args.tag,
        out_dir=out_dir,
    )

    export_panel_B(
        nn_root=nn_root,
        tag=args.tag,
        out_dir=out_dir,
        max_fixations=args.max_fixations,
        cumtime_mode=args.cumtime_mode,
    )

    export_panel_EF(
        nn_root=nn_root,
        tag=args.tag,
        out_dir=out_dir,
    )

    export_panel_GH(
        nn_root=nn_root,
        tag=args.tag,
        out_dir=out_dir,
        denom=args.gh_denom,
    )

    # Continuous (unbinned) versions for regression with position as
    # a numeric predictor.  keep_first_n and max_fixations are set high
    # so no positions are collapsed or dropped.
    export_panel_EF(
        nn_root=nn_root,
        tag=args.tag,
        out_dir=out_dir,
        max_fixations=1000,
        keep_first_n=1000,
        filename_suffix="_continuous",
    )

    export_panel_GH(
        nn_root=nn_root,
        tag=args.tag,
        out_dir=out_dir,
        max_fixations=1000,
        keep_first_n=1000,
        denom=args.gh_denom,
        filename_suffix="_continuous",
    )

    print(f"\nDone. All CSVs written to {out_dir}")


if __name__ == "__main__":
    main()
