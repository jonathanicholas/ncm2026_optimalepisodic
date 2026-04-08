import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


def is_image_name(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parts = value.split("_")
    return len(parts) == 4 and all(len(p) > 0 for p in parts)


def load_behavior_values(beh_file: Path) -> pd.DataFrame:
    df = pd.read_csv(beh_file)
    # Keep only encoding/value rows with needed columns
    # Different pipelines in this repo may store the encoded reward on either
    # encoding event='value' or encoding event='image'. Accept both.
    enc_vals = df[(df.get("phase") == "encoding") & (df.get("event").isin(["value", "image"]))].copy()
    cols = [c for c in ["game", "image", "outcome"] if c in enc_vals.columns]
    if not set(["game", "image", "outcome"]).issubset(cols):
        raise ValueError(f"Behavioral file missing required columns: {beh_file}")
    enc_vals = enc_vals[cols + ["event"]].copy()
    # Ensure image/value present
    enc_vals = enc_vals.dropna(subset=["image", "outcome"])
    enc_vals["outcome"] = pd.to_numeric(enc_vals["outcome"], errors="coerce")
    enc_vals["game"] = pd.to_numeric(enc_vals["game"], errors="coerce")
    enc_vals = enc_vals.dropna(subset=["outcome", "game"]).copy()
    enc_vals["game"] = enc_vals["game"].astype(int)
    # If both event types are present, prefer the 'value' row.
    enc_vals["event_priority"] = (enc_vals["event"].astype(str) != "value").astype(int)
    enc_vals = enc_vals.sort_values(["game", "image", "event_priority"], kind="mergesort")
    enc_vals = enc_vals.drop_duplicates(subset=["game", "image"], keep="first").copy()
    enc_vals = enc_vals.drop(columns=["event", "event_priority"], errors="ignore")
    return enc_vals


def load_fixations(fix_file: Path) -> pd.DataFrame:
    df = pd.read_csv(fix_file)
    # Filter to choice fixations only
    df = df[(df.get("phase") == "choice") & (df.get("event") == "choice")].copy()
    # Keep only valid image names in roi_content
    df = df[df.get("roi_content", pd.Series(dtype=str)).apply(is_image_name)].copy()

    # Required columns
    required = [
        "game",
        "trial_number",
        "option",
        "choice",
        "roi_content",
        "fix_start",
        "fix_end",
        "fix_duration_bounded",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Fixation file missing required columns {missing}: {fix_file}")

    # Optional columns from fixation dataframe that we propagate
    optional = [c for c in ["correct", "rt"] if c in df.columns]

    # Clean/select
    df = df[required + optional].copy()
    # Types
    df["game"] = pd.to_numeric(df["game"], errors="coerce").astype("Int64")
    df["trial_number"] = pd.to_numeric(df["trial_number"], errors="coerce").astype("Int64")
    df["fix_start"] = pd.to_numeric(df["fix_start"], errors="coerce")
    df["fix_end"] = pd.to_numeric(df["fix_end"], errors="coerce")
    df["fix_duration_bounded"] = pd.to_numeric(df["fix_duration_bounded"], errors="coerce")
    if "correct" in df.columns:
        df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    if "rt" in df.columns:
        df["rt"] = pd.to_numeric(df["rt"], errors="coerce")

    df = df.dropna(subset=["game", "trial_number", "fix_start"]).copy()
    df["game"] = df["game"].astype(int)
    df["trial_number"] = df["trial_number"].astype(int)

    # Sort by trial time
    df = df.sort_values(["game", "trial_number", "fix_start"]).rename(
        columns={"roi_content": "image", "fix_duration_bounded": "fixation_duration"}
    )
    return df


def combine_consecutive_fixations(df: pd.DataFrame) -> pd.DataFrame:
    # Combine consecutive fixations to the same image within each trial
    out_rows = []
    for (game, trial), sub in df.groupby(["game", "trial_number"], sort=False):
        # Maintain chronological order
        sub = sub.sort_values("fix_start")
        prev_image: Optional[str] = None
        combined = []
        for _, r in sub.iterrows():
            cur_image = r["image"]
            if prev_image is None or cur_image != prev_image:
                combined.append(
                    {
                        "game": game,
                        "trial_number": trial,
                        "option": r["option"],
                        "choice": r["choice"],
                        "correct": r.get("correct") if "correct" in sub.columns else None,
                        "rt": r.get("rt") if "rt" in sub.columns else None,
                        "image": cur_image,
                        "fix_start": r["fix_start"],
                        "fix_end": r["fix_end"],
                        "fixation_duration": r["fixation_duration"],
                    }
                )
            else:
                # Merge into previous
                combined[-1]["fix_end"] = max(combined[-1]["fix_end"], r["fix_end"])
                combined[-1]["fixation_duration"] += r["fixation_duration"]
            prev_image = cur_image

        # Assign sequential fixation count within trial after combination
        for i, item in enumerate(combined, start=1):
            item["fixation_count"] = i
            out_rows.append(item)

    if not out_rows:
        return pd.DataFrame(
            columns=[
                "game",
                "trial_number",
                "option",
                "choice",
                "correct",
                "rt",
                "image",
                "fix_start",
                "fix_end",
                "fixation_duration",
                "fixation_count",
            ]
        )
    return pd.DataFrame(out_rows)


def compute_relevance(image: str, option: str) -> int:
    if not isinstance(image, str) or not isinstance(option, str):
        return 0
    return 1 if option in image.split("_") else 0


def parse_buffer_from_filename(path: Path) -> Optional[int]:
    m = re.search(r"buffer_(\d+)\.csv$", path.name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def process_subject(subid: str, base_dir: Path) -> Tuple[pd.DataFrame, Optional[int]]:
    data_dir = base_dir / "data" / subid

    beh_file = data_dir / f"{subid}_MAIN_logfile_7.csv"
    fix_file = data_dir / f"{subid}_fixations_df_original_buffer_50.csv"

    if not beh_file.exists():
        return pd.DataFrame(), None
    if not fix_file.exists():
        return pd.DataFrame(), None

    beh_vals = load_behavior_values(beh_file)
    fix = load_fixations(fix_file)
    if fix.empty:
        return pd.DataFrame(), None

    fix_comb = combine_consecutive_fixations(fix)

    # Merge reward by (game, image)
    merged = fix_comb.merge(beh_vals, how="left", on=["game", "image"])  # adds outcome
    merged = merged.rename(columns={"outcome": "reward"})
    merged["subject_id"] = subid
    merged["relevance"] = merged.apply(lambda r: compute_relevance(r["image"], r["option"]), axis=1)

    # Final selection
    merged = merged[
        [
            "subject_id",
            "game",
            "trial_number",
            "option",
            "choice",
            "correct",
            "rt",
            "image",
            "reward",
            "relevance",
            "fixation_duration",
            "fixation_count",
        ]
    ].sort_values(["subject_id", "game", "trial_number", "fixation_count"]).reset_index(drop=True)
    buffer_size = parse_buffer_from_filename(fix_file)
    return merged, buffer_size


def find_subject_ids(data_root: Path, only_subjects: Optional[List[str]] = None) -> List[str]:
    if only_subjects:
        return [s for s in only_subjects if (data_root / s).is_dir()]
    subs = []
    for p in sorted(data_root.iterdir()):
        if p.is_dir() and p.name.isdigit():
            subs.append(p.name)
    return subs


def main():
    parser = argparse.ArgumentParser(description="Prepare clean choice fixation dataset across subjects.")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2]),
        help="Project base directory (default: repository root)",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        help="Optional list of subject IDs to include (defaults to all numeric subfolders in data/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: <base-dir>/output/choice_fixations_clean.csv)",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    data_root = base_dir / "data"
    specified_output = Path(args.output) if args.output else None

    subject_ids = find_subject_ids(data_root, args.subjects)
    all_frames = []
    buffers: List[int] = []
    for sid in subject_ids:
        try:
            df, buf = process_subject(sid, base_dir)
            if not df.empty:
                all_frames.append(df)
                if buf is not None:
                    buffers.append(buf)
        except Exception as e:
            # Continue on per-subject errors but log to console
            print(f"Warning: failed to process subject {sid}: {e}")

    if not all_frames:
        print("No data found to write.")
        return

    full = pd.concat(all_frames, ignore_index=True)
    # Decide output path
    if specified_output is not None:
        output_path = specified_output
    else:
        out_dir = base_dir / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        uniq = sorted(set(buffers))
        if len(uniq) == 1:
            output_path = out_dir / f"choice_fixations_clean_buffer_{uniq[0]}.csv"
        elif len(uniq) == 0:
            output_path = out_dir / "choice_fixations_clean.csv"
        else:
            output_path = out_dir / "choice_fixations_clean_buffer_mixed.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(output_path, index=False)
    print(f"Wrote {len(full)} rows to {output_path}")


if __name__ == "__main__":
    main()
