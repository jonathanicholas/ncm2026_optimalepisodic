"""Prepare choice-phase fixation data for adapted aDDM modeling.

Reads per-subject fixation CSVs from output/<subid>/ and produces cleaned
fixation CSVs with one row per merged fixation during the choice phase.

Processing: filters to choice-phase item fixations (optionally including
center fixations via --with-center), merges consecutive same-item fixations,
infers item relevance from the option column, and assigns integer item indices.

Output per subject:
    output/<subid>/<subid>_fixations_for_modeling.csv
    (or _fixations_for_modeling_withcenter.csv if --with-center)

Columns: subject, phase, event, trial_number, game, option, choice, rt,
roi_content, item_index, is_relevant, fix_start, fix_end, fix_duration_bounded.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class CleaningConfig:
    """Configuration for fixation cleaning.

    Attributes
    ----------
    base_output_dir : Path
        Path to the `output` directory containing per-subject folders.
        Cleaned output files are written here.
    base_input_dir : Path or None
        If set, raw fixation CSV files are read from this directory
        instead of ``base_output_dir``.  Subject discovery also uses
        this directory when set.
    """

    base_output_dir: Path
    base_input_dir: Optional[Path] = None

    @property
    def read_dir(self) -> Path:
        """Directory used for reading raw fixation CSVs and subject discovery."""
        return self.base_input_dir if self.base_input_dir is not None else self.base_output_dir


def _is_item_series(roi_series: pd.Series) -> pd.Series:
    """Return a boolean mask indicating which roi_content entries are items.

    Items are defined as non-null strings that are not 'fixation'/'none' and
    contain exactly three underscores (four feature tokens).
    """

    mask = roi_series.notna()
    roi_lower = roi_series.str.lower().fillna("")
    mask &= ~roi_lower.isin(["fixation", "none"])
    # Four features -> three underscores
    mask &= roi_series.str.count("_") == 3
    return mask


def _is_center_series(roi_series: pd.Series) -> pd.Series:
    """Return a boolean mask for center-ROI fixations (roi_content == 'fixation')."""

    roi_lower = roi_series.astype(str).str.lower().fillna("")
    return roi_lower.eq("fixation")


def _merge_consecutive_fixations(df_trial: pd.DataFrame) -> pd.DataFrame:
    """Merge consecutive fixations to the same item within a single trial.

    Parameters
    ----------
    df_trial : DataFrame
        Rows for a single trial (single trial_number), already filtered to
        choice phase and item fixations.

    Returns
    -------
    DataFrame
        DataFrame with one row per merged fixation, sorted by fix_start.
    """

    if df_trial.empty:
        return df_trial

    df = df_trial.sort_values("fix_start").reset_index(drop=True)

    # New group whenever roi_content changes from the previous row.
    change = df["roi_content"] != df["roi_content"].shift(1)
    group_id = change.cumsum()

    agg = df.groupby(group_id, as_index=False).agg(
        {
            # trial-level or event-level fields (should be identical within trial)
            "phase": "first",
            "event": "first",
            "trial_number": "first",
            "game": "first",
            "option": "first",
            "choice": "first",
            "rt": "first",
            "roi_content": "first",
            # timing fields
            "fix_start": "min",
            "fix_duration_bounded": "sum",
            # keep these as auxiliary (optional)
            "fix_duration_full": "sum",
            "fix_end": "max",
        }
    )

    # Define merged fixation end as start + total bounded duration.
    agg["fix_end_merged"] = agg["fix_start"] + agg["fix_duration_bounded"]

    return agg.sort_values("fix_start").reset_index(drop=True)


def _infer_relevance(row: pd.Series) -> int:
    """Infer item relevance from option and roi_content.

    Returns 1 if the option feature appears as one of the four feature tokens
    in roi_content, 0 otherwise.
    """

    option = row.get("option")
    roi = row.get("roi_content")

    if not isinstance(option, str) or not isinstance(roi, str):
        return 0

    # roi_content is of the form Feature1_Feature2_Feature3_Feature4
    parts = roi.split("_")
    return int(option in parts)


def clean_subject_fixations(subid: str, config: CleaningConfig, *, with_center: bool = False) -> Path:
    """Load and clean fixations for a single subject.

    Parameters
    ----------
    subid : str
        Subject identifier, e.g. '101'.
    config : CleaningConfig
        Configuration with base_output_dir pointing to `output`.

    Returns
    -------
    Path
        Path to the written cleaned CSV file.
    """

    input_sub_dir = config.read_dir / subid
    input_path = input_sub_dir / f"{subid}_fixations_df_original_buffer_50.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found for subject {subid}: {input_path}")

    df = pd.read_csv(input_path)

    # Filter to choice phase and choice events.
    mask_choice_phase = df["phase"] == "choice"
    mask_choice_event = df["event"] == "choice"
    df_choice = df[mask_choice_phase & mask_choice_event].copy()

    if df_choice.empty:
        raise ValueError(f"No choice-phase choice events found for subject {subid}")

    # Keep item fixations, and optionally center fixations.
    item_mask = _is_item_series(df_choice["roi_content"])
    if bool(with_center):
        center_mask = _is_center_series(df_choice["roi_content"])
        df_keep = df_choice[item_mask | center_mask].copy()
    else:
        df_keep = df_choice[item_mask].copy()

    if df_keep.empty:
        if bool(with_center):
            raise ValueError(f"No item/center fixations in choice phase for subject {subid}")
        raise ValueError(f"No item fixations in choice phase for subject {subid}")

    # Merge consecutive fixations to the same item within each trial.
    merged_list: List[pd.DataFrame] = []
    for trial_num, df_trial in df_keep.groupby("trial_number"):
        merged = _merge_consecutive_fixations(df_trial)
        merged_list.append(merged)

    df_merged = pd.concat(merged_list, ignore_index=True)

    # Mark center fixations and infer relevance for item fixations.
    df_merged["is_center"] = _is_center_series(df_merged["roi_content"]).astype(int)
    df_merged["is_relevant"] = 0
    item_rows = df_merged["is_center"].to_numpy(dtype=int) == 0
    if np.any(item_rows):
        df_merged.loc[item_rows, "is_relevant"] = df_merged.loc[item_rows].apply(_infer_relevance, axis=1)

    # Assign item_index within each game for *items only*; center uses -1.
    df_merged["item_index"] = -1
    for game, df_g in df_merged[item_rows].groupby("game"):
        codes, _uniques = pd.factorize(df_g["roi_content"].astype(str))
        df_merged.loc[df_g.index, "item_index"] = codes.astype(int)

    # Standardize column names / select a core set for modeling.
    df_merged["subject"] = subid

    # Ensure we have a clean fixation end column.
    df_merged["fix_end"] = df_merged["fix_end_merged"]

    cols_for_model = [
        "subject",
        "phase",
        "event",
        "trial_number",
        "game",
        "option",
        "choice",
        "rt",
        "roi_content",
        "item_index",
        "is_center",
        "is_relevant",
        "fix_start",
        "fix_end",
        "fix_duration_bounded",
    ]

    df_out = df_merged[cols_for_model].copy().sort_values(
        ["trial_number", "fix_start"]
    )

    output_sub_dir = config.base_output_dir / subid
    output_sub_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_sub_dir / f"{subid}_fixations_for_modeling_withcenter.csv"
        if bool(with_center)
        else output_sub_dir / f"{subid}_fixations_for_modeling.csv"
    )
    df_out.to_csv(output_path, index=False)

    return output_path


def find_subject_ids(config: CleaningConfig) -> List[str]:
    """Discover subject IDs based on available fixation CSV files."""

    subject_ids: List[str] = []
    for sub_dir in sorted(config.read_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        name = sub_dir.name
        csv_path = sub_dir / f"{name}_fixations_df_original_buffer_50.csv"
        if csv_path.exists():
            subject_ids.append(name)
    return subject_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clean choice-phase fixation data for adapted aDDM modeling. "
            "By default, processes all subjects found under the output directory."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Path to the output directory containing per-subject folders.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help=(
            "If set, read raw fixation CSV files from this directory instead of "
            "--output-dir. Subject discovery also uses this directory. "
            "Cleaned output files are still written under --output-dir."
        ),
    )
    parser.add_argument(
        "--subject",
        "-s",
        type=str,
        default=None,
        help=(
            "Optional subject ID to process (e.g., '101'). If omitted, all "
            "subjects with matching fixation CSVs are processed."
        ),
    )

    parser.add_argument(
        "--with-center",
        action="store_true",
        help=(
            "If set, include center-ROI fixations (roi_content=='fixation') and write "
            "<subid>_fixations_for_modeling_withcenter.csv (items use item_index>=0; center uses -1)."
        ),
    )

    args = parser.parse_args()
    base_output_dir = Path(args.output_dir).resolve()
    base_input_dir = Path(args.input_dir).resolve() if args.input_dir is not None else None
    config = CleaningConfig(base_output_dir=base_output_dir, base_input_dir=base_input_dir)

    if args.subject is not None:
        subject_ids = [args.subject]
    else:
        subject_ids = find_subject_ids(config)

    if not subject_ids:
        raise SystemExit(f"No subject fixation CSVs found under {base_output_dir}")

    print(f"Found {len(subject_ids)} subject(s): {', '.join(subject_ids)}")

    for subid in subject_ids:
        try:
            out_path = clean_subject_fixations(subid, config, with_center=bool(args.with_center))
        except Exception as exc:  # pragma: no cover - simple CLI reporting
            print(f"[ERROR] Failed to process subject {subid}: {exc}")
            continue
        print(f"[OK] Wrote cleaned fixations for subject {subid} to {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
