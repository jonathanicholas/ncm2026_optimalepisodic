"""Build the long-form per-candidate CSVs for the next-fixation conditional
logit. Produces four datasets under output/next_fixation/:

  next_fixation_long_human.csv               humans
  next_fixation_long_rnn_input5_500k.csv     prior-memory network (500k trials)
  next_fixation_long_walk_ring_noisy_10x.csv adjacent null oracle (10x resampled)
  next_fixation_long_random_10x.csv          uniform-random null oracle (10x resampled)

Usage (from repo root):
  python metarnn/next_fixation/build_next_fixation_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "lib"))

from data_loaders import load_human, load_rnn, RNNLoaderConfig, summarize  # noqa: E402
from candidate_features import build_next_fixation_long  # noqa: E402
from lesion_oracles import walk_ring_noisy, random_oracle  # noqa: E402

OUT_DIR = REPO_ROOT / "output" / "next_fixation"


def write(df, name: str) -> None:
    if len(df) == 0:
        print(f"[skip] {name} produced 0 rows")
        return
    path = OUT_DIR / f"next_fixation_long_{name}.csv"
    df.to_csv(path, index=False)
    print(f"  wrote {path}  rows={len(df)}  events={df['event_id'].nunique()}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== humans ===")
    t_h, f_h = load_human()
    summarize(t_h, f_h, "human")
    write(build_next_fixation_long(t_h, f_h), "human")

    print("=== prior-memory network (100k trials) ===")
    t_n, f_n = load_rnn(RNNLoaderConfig(
        variant="input5", n_synthetic_subjects=1,
        trials_per_subject=100000, seed=0))
    summarize(t_n, f_n, "rnn_input5_500k")
    write(build_next_fixation_long(t_n, f_n), "rnn_input5_500k")

    print("=== adjacent null oracle (10x) ===")
    t_m, f_m = walk_ring_noisy(t_h, f_h, seed=0, n_repeats=10)
    write(build_next_fixation_long(t_m, f_m), "walk_ring_noisy_10x")

    print("=== uniform-random null oracle (10x) ===")
    t_r, f_r = random_oracle(t_h, f_h, seed=0, n_repeats=10)
    write(build_next_fixation_long(t_r, f_r), "random_10x")


if __name__ == "__main__":
    main()
