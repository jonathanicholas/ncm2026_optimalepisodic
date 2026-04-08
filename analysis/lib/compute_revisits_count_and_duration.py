"""Quantify all-fixation, first-fixation, and revisit counts/durations (relevant vs irrelevant; humans vs NN).

Applies to choice-phase fixations in:
- output/<subid>/<subid>_fixations_for_modeling.csv

Definitions
-----------
Within each trial (game x trial_number x subject), for each item_index:
- The first fixation to that item is not a revisit.
- Any subsequent fixation(s) to the same item are counted as revisit fixation events.

We compute, per subject:
1) Mean all-fixation count per trial (event-level)
    - allfix_count_relevant_per_trial
    - allfix_count_irrelevant_per_trial

2) Mean duration of all-fixation events
    - allfix_duration_mean_relevant
    - allfix_duration_mean_irrelevant

3) Mean first-fixation count per trial (event-level)
    - firstfix_count_relevant_per_trial
    - firstfix_count_irrelevant_per_trial

4) Mean duration of first-fixation events
    - firstfix_duration_mean_relevant
    - firstfix_duration_mean_irrelevant

5) Mean revisit count per trial (event-level)
    - revisit_count_relevant_per_trial
    - revisit_count_irrelevant_per_trial

6) Mean duration of revisit fixation events
    - revisit_duration_mean_relevant
    - revisit_duration_mean_irrelevant

7) Time-proportion summaries
        - Within first-fixations: proportion of first-fixation time on relevant vs irrelevant items
            (firstfix_prop_time_relevant, firstfix_prop_time_irrelevant)
        - Within revisits: proportion of revisit time on relevant vs irrelevant items
            (revisit_prop_time_relevant, revisit_prop_time_irrelevant)
        - Share of all item-looking time coming from first vs revisit fixations
            (time_share_firstfix, time_share_revisit)

Then we aggregate across subjects separately for humans and NN.

Outputs
-------
Written to --out-dir (default: <base-dir>/output/eyegaze/stats):
- revisits_count_and_duration_by_subject[_TAG].csv
- revisits_count_and_duration_summary[_TAG].csv

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    base_dir: Path
    out_dir: Path
    excluded_subjects: Tuple[str, ...]
    nn_base_dir: Path
    nn_excluded_subjects: Tuple[str, ...]
    nn_out_dir: Path
    output_tag: str
    time_unit_label: str
    nn_time_unit_label: str
    n_boot: int
    boot_seed: int


def _tagged_filename(name: str, *, tag: str, ext: str) -> str:
    tag = str(tag).strip()
    suffix = f"_{tag}" if tag else ""
    return f"{name}{suffix}.{ext.lstrip('.')}"


def _list_numeric_subdirs(parent: Path) -> List[str]:
    if not parent.exists():
        return []
    return sorted([p.name for p in parent.iterdir() if p.is_dir() and p.name.isdigit()])


def _list_subject_ids(output_dir: Path, excluded_subjects: Iterable[str]) -> List[str]:
    excluded = set(str(s) for s in excluded_subjects)
    return [s for s in _list_numeric_subdirs(output_dir) if s not in excluded]


def load_fixations_for_subject(output_dir: Path, subid: str) -> pd.DataFrame:
    fix_file = output_dir / subid / f"{subid}_fixations_for_modeling.csv"
    if not fix_file.exists():
        raise FileNotFoundError(f"Missing fixation file: {fix_file}")

    df = pd.read_csv(fix_file)

    required = {
        "subject",
        "game",
        "trial_number",
        "item_index",
        "fix_start",
        "fix_duration_bounded",
        "is_relevant",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{fix_file} missing required columns: {missing}")

    # Restrict to choice if phase exists.
    if "phase" in df.columns:
        df = df[df["phase"].astype(str).str.lower() == "choice"].copy()

    for c in ["game", "trial_number", "item_index", "fix_start", "fix_duration_bounded", "is_relevant"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["subject", "game", "trial_number", "item_index", "fix_start", "fix_duration_bounded", "is_relevant"]).copy()
    df["subject"] = df["subject"].astype(str)
    df["game"] = df["game"].astype(int)
    df["trial_number"] = df["trial_number"].astype(int)
    df["item_index"] = df["item_index"].astype(int)
    df["is_relevant"] = df["is_relevant"].astype(int)

    return df


def compute_subject_metrics(df: pd.DataFrame) -> dict:
    df = df.sort_values(["game", "trial_number", "fix_start"]).copy()

    # First fixation to item within trial
    df["is_first_fix_to_item"] = (
        df.groupby(["game", "trial_number", "item_index"]).cumcount() == 0
    )
    df["is_revisit_fixation"] = ~df["is_first_fix_to_item"]

    # Trial count
    n_trials = int(df[["game", "trial_number"]].drop_duplicates().shape[0])

    out = {
        "n_trials": n_trials,
        "n_fixations": int(len(df)),
    }

    # All-fixation metrics (event-coded)
    for rel, lab in [(1, "relevant"), (0, "irrelevant")]:
        sub = df[(df["is_relevant"] == rel)].copy()

        # Count all fixation events per trial
        if n_trials > 0:
            out[f"allfix_count_{lab}_per_trial"] = float(len(sub) / n_trials)
        else:
            out[f"allfix_count_{lab}_per_trial"] = np.nan

        # Mean duration of all fixation events
        if len(sub) > 0:
            out[f"allfix_duration_mean_{lab}"] = float(sub["fix_duration_bounded"].mean())
        else:
            out[f"allfix_duration_mean_{lab}"] = np.nan

        out[f"allfix_count_{lab}"] = int(len(sub))

    # First-fixation metrics (item-level first visit within trial, event-coded)
    for rel, lab in [(1, "relevant"), (0, "irrelevant")]:
        sub = df[(df["is_relevant"] == rel) & (df["is_first_fix_to_item"])].copy()

        # Count first-fixation events per trial
        if n_trials > 0:
            out[f"firstfix_count_{lab}_per_trial"] = float(len(sub) / n_trials)
        else:
            out[f"firstfix_count_{lab}_per_trial"] = np.nan

        # Mean duration of first-fixation events
        if len(sub) > 0:
            out[f"firstfix_duration_mean_{lab}"] = float(sub["fix_duration_bounded"].mean())
        else:
            out[f"firstfix_duration_mean_{lab}"] = np.nan

        out[f"firstfix_count_{lab}"] = int(len(sub))

    for rel, lab in [(1, "relevant"), (0, "irrelevant")]:
        sub = df[(df["is_relevant"] == rel) & (df["is_revisit_fixation"])].copy()

        # Count revisits per trial (event-level)
        if n_trials > 0:
            out[f"revisit_count_{lab}_per_trial"] = float(len(sub) / n_trials)
        else:
            out[f"revisit_count_{lab}_per_trial"] = np.nan

        # Mean duration of revisit fixation events
        if len(sub) > 0:
            out[f"revisit_duration_mean_{lab}"] = float(sub["fix_duration_bounded"].mean())
        else:
            out[f"revisit_duration_mean_{lab}"] = np.nan

        # Raw counts (sometimes useful)
        out[f"revisit_count_{lab}"] = int(len(sub))

    # ------------------------------------------------------------------
    # Time-proportion summaries
    # ------------------------------------------------------------------
    # Total item-looking time (all fixations)
    total_time_all = float(df["fix_duration_bounded"].sum()) if len(df) else 0.0

    first_df = df[df["is_first_fix_to_item"]].copy()
    revisit_df = df[df["is_revisit_fixation"]].copy()

    first_time_total = float(first_df["fix_duration_bounded"].sum()) if len(first_df) else 0.0
    revisit_time_total = float(revisit_df["fix_duration_bounded"].sum()) if len(revisit_df) else 0.0

    # Within-type relevant/irrelevant time splits
    first_time_rel = float(first_df.loc[first_df["is_relevant"] == 1, "fix_duration_bounded"].sum()) if len(first_df) else 0.0
    first_time_irr = float(first_df.loc[first_df["is_relevant"] == 0, "fix_duration_bounded"].sum()) if len(first_df) else 0.0
    revisit_time_rel = float(revisit_df.loc[revisit_df["is_relevant"] == 1, "fix_duration_bounded"].sum()) if len(revisit_df) else 0.0
    revisit_time_irr = float(revisit_df.loc[revisit_df["is_relevant"] == 0, "fix_duration_bounded"].sum()) if len(revisit_df) else 0.0

    first_denom = first_time_rel + first_time_irr
    if first_denom > 0:
        out["firstfix_prop_time_relevant"] = float(first_time_rel / first_denom)
        out["firstfix_prop_time_irrelevant"] = float(first_time_irr / first_denom)
    else:
        out["firstfix_prop_time_relevant"] = np.nan
        out["firstfix_prop_time_irrelevant"] = np.nan

    revisit_denom = revisit_time_rel + revisit_time_irr
    if revisit_denom > 0:
        out["revisit_prop_time_relevant"] = float(revisit_time_rel / revisit_denom)
        out["revisit_prop_time_irrelevant"] = float(revisit_time_irr / revisit_denom)
    else:
        out["revisit_prop_time_relevant"] = np.nan
        out["revisit_prop_time_irrelevant"] = np.nan

    # Share of total item-looking time (all-fix) coming from first vs revisits
    if total_time_all > 0:
        out["time_share_firstfix"] = float(first_time_total / total_time_all)
        out["time_share_revisit"] = float(revisit_time_total / total_time_all)
    else:
        out["time_share_firstfix"] = np.nan
        out["time_share_revisit"] = np.nan

    return out


def bootstrap_ci(values: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float]:
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n = len(v)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = float(rng.choice(v, size=n, replace=True).mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def summarize_across_subjects(by_sub: pd.DataFrame, *, n_boot: int, seed: int) -> pd.DataFrame:
    metrics = [
        "allfix_count_relevant_per_trial",
        "allfix_count_irrelevant_per_trial",
        "allfix_duration_mean_relevant",
        "allfix_duration_mean_irrelevant",
        "firstfix_count_relevant_per_trial",
        "firstfix_count_irrelevant_per_trial",
        "firstfix_duration_mean_relevant",
        "firstfix_duration_mean_irrelevant",
        "revisit_count_relevant_per_trial",
        "revisit_count_irrelevant_per_trial",
        "revisit_duration_mean_relevant",
        "revisit_duration_mean_irrelevant",
        "firstfix_prop_time_relevant",
        "firstfix_prop_time_irrelevant",
        "revisit_prop_time_relevant",
        "revisit_prop_time_irrelevant",
        "time_share_firstfix",
        "time_share_revisit",
    ]
    rows = []
    for m in metrics:
        vals = pd.to_numeric(by_sub[m], errors="coerce").to_numpy(dtype=float)
        mean = float(np.nanmean(vals))
        lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "metric": m,
                "mean": mean,
                "ci_lo": lo,
                "ci_hi": hi,
                "n_subjects": int(np.isfinite(vals).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    default_base_dir = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Compute first-fixation and revisit counts/durations (relevant vs irrelevant)")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(default_base_dir),
        help="Dataset root containing output/<subid>/ (default: repo root).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Where to write CSVs (default: <base-dir>/output/eyegaze/stats).",
    )
    parser.add_argument(
        "--exclude-subjects",
        nargs="*",
        default=["107", "131"],
        help="Subject IDs to exclude (default: 107 131). For NN sims you likely want an empty list.",
    )
    parser.add_argument(
        "--nn-base-dir",
        type=str,
        default="",
        help=(
            "Optional NN dataset root (containing output/<subid>/). "
            "If provided, the script will compute both humans (base-dir) and NN."
        ),
    )
    parser.add_argument(
        "--nn-exclude-subjects",
        nargs="*",
        default=[],
        help="Subject IDs to exclude for NN dataset (default: none).",
    )
    parser.add_argument(
        "--nn-out-dir",
        type=str,
        default="",
        help="Where to write NN CSVs (default: same as --out-dir).",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default="",
        help="Optional suffix added to output filenames.",
    )
    parser.add_argument(
        "--time-unit-label",
        type=str,
        default="ms",
        help="Label for time units on the duration panel (e.g., ms or steps).",
    )
    parser.add_argument(
        "--nn-time-unit-label",
        type=str,
        default="",
        help=(
            "Optional label for NN time units on the NN duration panel (e.g., 'steps'). "
            "Defaults to --time-unit-label if not provided."
        ),
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=2000,
        help="Bootstrap draws for 95% CI across subjects.",
    )
    parser.add_argument(
        "--boot-seed",
        type=int,
        default=0,
        help="RNG seed for bootstrapping.",
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if str(args.out_dir).strip() else (base_dir / "output" / "eyegaze" / "stats")
    out_dir.mkdir(parents=True, exist_ok=True)
    nn_out_dir = Path(args.nn_out_dir).resolve() if str(args.nn_out_dir).strip() else out_dir

    config = Config(
        base_dir=base_dir,
        out_dir=out_dir,
        excluded_subjects=tuple(str(s) for s in (args.exclude_subjects or [])),
        nn_base_dir=(Path(args.nn_base_dir).resolve() if str(args.nn_base_dir).strip() else Path("")),
        nn_excluded_subjects=tuple(str(s) for s in (args.nn_exclude_subjects or [])),
        nn_out_dir=nn_out_dir,
        output_tag=str(args.output_tag),
        time_unit_label=str(args.time_unit_label),
        nn_time_unit_label=(str(args.nn_time_unit_label).strip() or str(args.time_unit_label)),
        n_boot=int(args.n_boot),
        boot_seed=int(args.boot_seed),
    )

    def _run_dataset(dataset_base: Path, excluded_subjects: Tuple[str, ...], tag: str, *, dest_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Determine which subdirectory holds fixation_for_modeling CSVs.
        # Human layout: data/{SUBID}/; NN sim layout: output/{SUBID}/
        output_dir = dataset_base / "data"
        subids = _list_subject_ids(output_dir, excluded_subjects)
        # Verify fixation files actually exist in data/; if not, try output/
        if subids:
            test_file = output_dir / subids[0] / f"{subids[0]}_fixations_for_modeling.csv"
            if not test_file.exists():
                output_dir = dataset_base / "output"
                subids = _list_subject_ids(output_dir, excluded_subjects)
        else:
            output_dir = dataset_base / "output"
            subids = _list_subject_ids(output_dir, excluded_subjects)
        if len(subids) == 0:
            raise ValueError(f"No subject directories found under {output_dir}")

        dest_dir.mkdir(parents=True, exist_ok=True)

        rows_local = []
        for subid in subids:
            df = load_fixations_for_subject(output_dir, subid)
            m = compute_subject_metrics(df)
            m["subject"] = str(subid)
            rows_local.append(m)

        by_sub_local = pd.DataFrame(rows_local)
        out_by_sub_local = dest_dir / _tagged_filename(
            "revisits_count_and_duration_by_subject", tag=tag, ext="csv"
        )
        by_sub_local.to_csv(out_by_sub_local, index=False)

        summary_local = summarize_across_subjects(by_sub_local, n_boot=config.n_boot, seed=config.boot_seed)
        out_summary_local = dest_dir / _tagged_filename(
            "revisits_count_and_duration_summary", tag=tag, ext="csv"
        )
        summary_local.to_csv(out_summary_local, index=False)

        print(f"Wrote {out_by_sub_local}")
        print(f"Wrote {out_summary_local}")
        return by_sub_local, summary_local

    # Always run the base-dir dataset (humans)
    _run_dataset(config.base_dir, config.excluded_subjects, "human", dest_dir=config.out_dir)

    # Optionally run NN
    if str(config.nn_base_dir) and config.nn_base_dir.exists():
        nn_tag = f"nn{('_' + config.output_tag) if config.output_tag else ''}"
        _run_dataset(config.nn_base_dir, config.nn_excluded_subjects, nn_tag, dest_dir=config.nn_out_dir)
    else:
        print("Note: --nn-base-dir not provided; skipping NN computation.")


if __name__ == "__main__":
    main()
