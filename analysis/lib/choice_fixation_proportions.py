"""Compute fixation proportions by relevance and valence during choice trials.

Data sources per subject id (three-digit string):
  Fixations: output/<id>/<id>_fixations_df_original_buffer_50.csv
  Behavioral logfile: data/<id>/<id>_MAIN_logfile_7.csv

Outputs (saved to --out-dir, default output/eyegaze/stats/):
  1. choice_fixation_relevance_subject_means_relevant_only_{metric}.csv
  2. choice_fixation_relsign4_relevant_subject_means_{metric}.csv
  3. choice_fixation_relevant_trial_deltas_long_{metric}.csv
  4. choice_fixation_irrelevant_trial_deltas_long_{metric}.csv

Valence: sign of `outcome` (>0 positive, <0 negative, ==0 neutral). Mapped per (game, image).

Relevance: Whether the trial's offered option token (column `option` from the choice
phase rows) appears within the image name (substring match). If yes => relevant.

Decision: Column `choice` in the fixation rows (1=take, 2=leave).
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from recalled_valence import build_recalled_valence_map


@dataclass
class TrialImageFixation:
    subject: str
    game: int
    trial_number: int
    image: str
    decision: int  # 1=take, 2=leave
    option_token: Optional[str]
    valence: Optional[str]
    outcome_value: Optional[float]
    relevance: bool
    fixation_time: float  # duration or count per image
    total_image_fix_time: float  # total across images (duration or count)
    measure: str  # 'duration' or 'count'

    @property
    def fixation_proportion(self) -> float:
        if self.total_image_fix_time <= 0:
            return np.nan
        return self.fixation_time / self.total_image_fix_time


def find_subjects(output_root: str) -> List[str]:
    subjects = []
    for path in glob.glob(os.path.join(output_root, '*')):
        if os.path.isdir(path):
            base = os.path.basename(path)
            if base.isdigit() and len(base) == 3:
                subjects.append(base)
    subjects.sort()
    return subjects


def build_valence_map(log_df: pd.DataFrame, subject: str = None, data_dir: str = None) -> Dict[tuple, tuple]:
    """Return {(game, image): (valence_str, value)}.

    Uses recalled values when subject and data_dir are provided,
    falling back to true outcome for missing recalls.
    """
    if subject is not None and data_dir is not None:
        return build_recalled_valence_map(subject, data_dir)

    # True outcome only
    enc = log_df[(log_df['phase'] == 'encoding') & (log_df['event'] == 'image')]
    mapping: Dict[tuple, tuple] = {}
    for _, row in enc.iterrows():
        game = int(row['game'])
        image = str(row['image'])
        outcome = row.get('outcome')
        try:
            val = float(outcome)
        except (TypeError, ValueError):
            val = np.nan
        if np.isnan(val):
            valence = 'neutral'
        elif val > 0:
            valence = 'positive'
        elif val < 0:
            valence = 'negative'
        else:
            valence = 'neutral'
        mapping[(game, image)] = (valence, val)
    return mapping


def process_subject(subject: str, root: str, metric: str = 'duration') -> List[TrialImageFixation]:
    fix_path = os.path.join(root, 'data', subject, f'{subject}_fixations_df_original_buffer_50.csv')
    log_path = os.path.join(root, 'data', subject, f'{subject}_MAIN_logfile_7.csv')
    if not (os.path.exists(fix_path) and os.path.exists(log_path)):
        print(f'[WARN] Missing files for subject {subject}; skipping.')
        return []

    fix_df = pd.read_csv(fix_path)
    log_df = pd.read_csv(log_path)

    # Defensive lowercase normalization for columns if needed
    for col in ['phase', 'event']:
        if col in fix_df.columns:
            fix_df[col] = fix_df[col].astype(str).str.lower()
        if col in log_df.columns:
            log_df[col] = log_df[col].astype(str).str.lower()

    data_dir = os.path.join(root, 'data')
    valence_map = build_valence_map(log_df, subject=subject, data_dir=data_dir)

    choice_rows = fix_df[(fix_df['phase'] == 'choice') & (fix_df['event'] == 'choice')].copy()
    if choice_rows.empty:
        print(f'[WARN] No choice rows for subject {subject}.')
        return []

    results: List[TrialImageFixation] = []
    group_cols = ['game', 'trial_number']
    # Some files may store floats for these; coerce early.
    for c in group_cols + ['choice']:
        if c in choice_rows.columns:
            choice_rows.loc[:, c] = pd.to_numeric(choice_rows[c], errors='coerce')

    grouped = choice_rows.groupby(group_cols, sort=True)

    for (game, trial), grp in grouped:
        # Decision (choice) constant across rows; take first valid
        decision_vals = grp['choice'].dropna().unique()
        decision = int(decision_vals[0]) if len(decision_vals) else -1

        # Option token (offer) from 'option' column
        option_token = None
        if 'option' in grp.columns:
            opts = grp['option'].dropna().astype(str).unique()
            if len(opts):
                option_token = opts[0]

        # Image fixation filtering
        image_fix = grp[(~grp['roi_content'].isin(['fixation', 'none'])) & (grp['roi_content'].astype(str).str.contains('_'))]

        if image_fix.empty:
            continue  # Skip trials with no image fixations

        # Aggregate per-image either by duration or by count
        if metric == 'count':
            total_time = float(len(image_fix))
            per_image = image_fix.groupby('roi_content').size().astype(float)
        else:
            dur_col = 'fix_duration_bounded' if 'fix_duration_bounded' in image_fix.columns else 'fix_duration_full'
            total_time = image_fix[dur_col].sum()
            per_image = image_fix.groupby('roi_content')[dur_col].sum()

        for image_name, img_time in per_image.items():
            valence, outcome_val = valence_map.get((int(game), image_name), (None, None))
            relevance = False
            if option_token and isinstance(option_token, str):
                # Option token should match any component of image name
                relevance = option_token in image_name.split('_') or option_token in image_name

            results.append(
                TrialImageFixation(
                    subject=subject,
                    game=int(game),
                    trial_number=int(trial),
                    image=image_name,
                    decision=decision,
                    option_token=option_token,
                    valence=valence,
                    outcome_value=outcome_val,
                    relevance=relevance,
                    fixation_time=float(img_time),
                    total_image_fix_time=float(total_time),
                    measure=metric,
                )
            )
    return results


def build_dataframe(records: List[TrialImageFixation]) -> pd.DataFrame:
    # Exclude trials without a valid choice (decision not 1 or 2).
    filtered = [r for r in records if r.decision in (1, 2)]
    df = pd.DataFrame([{
        'subject': r.subject,
        'game': r.game,
        'trial_number': r.trial_number,
        'image': r.image,
        'decision': r.decision,
        'decision_label': 'take' if r.decision == 1 else 'leave',
        'option_token': r.option_token,
        'valence': r.valence,
        'outcome_value': r.outcome_value,
        'relevance': r.relevance,
        'fixation_time': r.fixation_time,
        'total_image_fix_time': r.total_image_fix_time,
        'fixation_proportion': r.fixation_proportion,
        'measure': r.measure,
        'rel_valence': f"{'relevant' if r.relevance else 'irrelevant'}_{r.valence}" if r.valence else None,
    } for r in filtered])
    return df


def _build_game_images_map(root: str, subject: str) -> Dict[int, List[tuple]]:
    """For a subject, return {game: [(image_name, valence_str), ...]} using recalled values."""
    data_dir = os.path.join(root, 'data')
    valence_map = build_recalled_valence_map(subject, data_dir)
    if not valence_map:
        return {}
    game_map: Dict[int, List[tuple]] = {}
    for (game, image), (valence, _val) in valence_map.items():
        game_map.setdefault(game, []).append((image, valence))
    return game_map


def aggregate_relsign4(image_df: pd.DataFrame, root: str) -> pd.DataFrame:
    """Aggregate per-image proportions into the four relsign4 categories per trial.

    Returns trial-level DataFrame with columns:
      subject, game, trial_number, decision, decision_label,
      rel_pos, rel_neg, irr_pos, irr_neg, total_image_fix_time,
      rel_pos_chance, rel_neg_chance, irr_pos_chance, irr_neg_chance
    Chance proportions are (#images in category)/6.
    """
    # Filter to positive/negative only (exclude neutral images if any)
    tmp = image_df[image_df['valence'].isin(['positive', 'negative'])].copy()
    # Determine per-image category flags for summing proportions
    tmp['category'] = tmp.apply(
        lambda r: (
            ('rel_' if r['relevance'] else 'irr_') + ('pos' if r['valence'] == 'positive' else 'neg')
        ), axis=1
    )
    # Build per-subject game->images map from logfile to count categories using all six images
    game_images_by_subject: Dict[str, Dict[int, List[tuple]]] = {}
    for subj in tmp['subject'].unique():
        game_images_by_subject[subj] = _build_game_images_map(root, subj)

    # Per-trial summed proportions per category
    prop_sum = (
        tmp.groupby(['subject', 'game', 'trial_number', 'category'])['fixation_proportion']
           .sum()
           .unstack('category')
           .fillna(0)
    )
    for col in ['rel_pos', 'rel_neg', 'irr_pos', 'irr_neg']:
        if col not in prop_sum.columns:
            prop_sum[col] = 0.0
    prop_sum = prop_sum[['rel_pos', 'rel_neg', 'irr_pos', 'irr_neg']]

    # Decision and option info (assume constant per trial; take first)
    decision_info = tmp.groupby(['subject', 'game', 'trial_number']).agg(
        decision=('decision', 'first'),
        decision_label=('decision_label', 'first'),
        total_image_fix_time=('total_image_fix_time', 'first'),
        option_token=('option_token', 'first')
    )

    # Compute per-trial category counts from full game image set
    counts_records = []
    for (subj, g, t), row in decision_info.iterrows():
        option = row['option_token'] if isinstance(row['option_token'], str) else None
        images = game_images_by_subject.get(subj, {}).get(int(g), [])
        rel_pos = rel_neg = irr_pos = irr_neg = 0
        if images:
            for img_name, valence in images:
                is_rel = bool(option) and (option in img_name.split('_') or option in img_name)
                if valence not in ('positive', 'negative'):
                    continue
                if is_rel and valence == 'positive':
                    rel_pos += 1
                elif is_rel and valence == 'negative':
                    rel_neg += 1
                elif (not is_rel) and valence == 'positive':
                    irr_pos += 1
                elif (not is_rel) and valence == 'negative':
                    irr_neg += 1
        counts_records.append({
            'subject': subj,
            'game': int(g),
            'trial_number': int(t),
            'rel_pos_count': rel_pos,
            'rel_neg_count': rel_neg,
            'irr_pos_count': irr_pos,
            'irr_neg_count': irr_neg,
        })
    img_counts = pd.DataFrame.from_records(counts_records).set_index(['subject', 'game', 'trial_number'])

    merged = decision_info.join(prop_sum).join(img_counts)
    # Compute chance proportions (#images in category)/6
    for cat in ['rel_pos', 'rel_neg', 'irr_pos', 'irr_neg']:
        merged[f'{cat}_chance'] = merged[f'{cat}_count'] / 6.0

    merged.reset_index(inplace=True)
    return merged


def main():
    parser = argparse.ArgumentParser(description='Compute choice fixation proportions by relevance and valence.')
    parser.add_argument('--root', default='.', help='Project root (default current).')
    parser.add_argument('--subjects', nargs='*', help='Optional specific subject IDs (e.g. 002 005).')
    parser.add_argument('--metric', choices=['duration','count'], default='duration', help='Aggregate by fixation duration or count (default: duration).')
    parser.add_argument('--out-dir', default=os.path.join('output', 'eyegaze', 'stats'), help='Output directory for CSV files.')
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if args.subjects:
        subjects = [s for s in args.subjects]
    else:
        subjects = find_subjects(os.path.join(root, 'data'))

    # Apply subject-level exclusion based on eyetracking quality (choice trials)
    exclusion_path = os.path.join(root, 'output', 'choice_trial_drop_overall.csv')
    excluded_subjects = set()
    if os.path.exists(exclusion_path):
        try:
            excl_df = pd.read_csv(exclusion_path)
            if 'subject_excluded_choice' in excl_df.columns and 'subject' in excl_df.columns:
                excluded_subjects = set(
                    excl_df.loc[excl_df['subject_excluded_choice'].astype(bool), 'subject'].astype(str)
                )
        except Exception:
            excluded_subjects = set()

    if excluded_subjects:
        before_n = len(subjects)
        subjects = [s for s in subjects if str(s) not in excluded_subjects]
        after_n = len(subjects)
        print(
            'Excluding subjects from choice fixation proportion analyses due to eyetracking quality (choice trials):',
            ', '.join(sorted(excluded_subjects)),
        )
        print(f'Subjects before exclusion: {before_n} | after exclusion: {after_n}')

    all_records: List[TrialImageFixation] = []
    for subj in subjects:
        recs = process_subject(subj, root, metric=args.metric)
        all_records.extend(recs)

    if not all_records:
        print('[ERROR] No records produced.')
        return

    df = build_dataframe(all_records)

    out_dir = os.path.join(root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ---- Output 1: Subject-level relevant-only fixation proportion means ----
    rel_irr_trial = (
        df.groupby(['subject', 'game', 'trial_number', 'relevance'])['fixation_proportion']
          .sum()
          .unstack('relevance')
          .rename(columns={True: 'relevant', False: 'irrelevant'})
          .fillna(0.0)
    )
    for col in ['relevant', 'irrelevant']:
        if col not in rel_irr_trial.columns:
            rel_irr_trial[col] = 0.0
    rel_irr_subj = (
        rel_irr_trial.groupby('subject')[['relevant', 'irrelevant']]
            .mean()
            .reset_index()
    )
    rel_irr_long = rel_irr_subj.melt(
        id_vars=['subject'],
        value_vars=['relevant', 'irrelevant'],
        var_name='relevance_label',
        value_name='mean_prop'
    )
    rel_only_long = rel_irr_long[rel_irr_long['relevance_label'] == 'relevant'].copy()
    rel_only_csv = os.path.join(out_dir, f'choice_fixation_relevance_subject_means_relevant_only_{args.metric}.csv')
    rel_only_long.to_csv(rel_only_csv, index=False)
    print(f'[INFO] Saved relevant-only subject means: {rel_only_csv} (n_rows={len(rel_only_long)})')

    # ---- Output 2: Relsign4 relevant subject means ----
    relsign4_df = aggregate_relsign4(df, root)
    subj_relsign4 = (
        relsign4_df.groupby(['subject', 'decision_label'])[['rel_pos', 'rel_neg', 'irr_pos', 'irr_neg']]
            .mean()
            .reset_index()
    )

    # Melt to long format and save relevant-only subject means
    subj_relsign4_long = subj_relsign4.melt(
        id_vars=['subject', 'decision_label'],
        value_vars=['rel_pos', 'rel_neg', 'irr_pos', 'irr_neg'],
        var_name='category', value_name='mean_prop')
    subj_relsign4_long['relevance'] = np.where(subj_relsign4_long['category'].str.startswith('rel_'), 'relevant', 'irrelevant')
    subj_relsign4_long['valence_label'] = np.where(subj_relsign4_long['category'].str.endswith('pos'), 'positive', 'negative')

    rel_subj_means = subj_relsign4_long[subj_relsign4_long['relevance'] == 'relevant'].copy()
    rel_subj_means_csv = os.path.join(out_dir, f'choice_fixation_relsign4_relevant_subject_means_{args.metric}.csv')
    rel_subj_means.to_csv(rel_subj_means_csv, index=False)
    print(f'[INFO] Saved relsign4 relevant subject means: {rel_subj_means_csv} (n_rows={len(rel_subj_means)})')

    # ---- Output 3: Trial-level relevant deltas from chance ----
    rel_trials = relsign4_df[['subject', 'game', 'trial_number', 'decision_label', 'rel_pos', 'rel_neg', 'rel_pos_chance', 'rel_neg_chance']].copy()
    rel_long = rel_trials.melt(
        id_vars=['subject', 'game', 'trial_number', 'decision_label'],
        value_vars=['rel_pos', 'rel_neg'],
        var_name='category', value_name='prop')
    chance_long = rel_trials.melt(
        id_vars=['subject', 'game', 'trial_number', 'decision_label'],
        value_vars=['rel_pos_chance', 'rel_neg_chance'],
        var_name='chance_col', value_name='chance')
    chance_long['category'] = chance_long['chance_col'].str.replace('_chance', '', regex=False)
    rel_long = rel_long.merge(
        chance_long[['subject', 'game', 'trial_number', 'decision_label', 'category', 'chance']],
        on=['subject', 'game', 'trial_number', 'decision_label', 'category'], how='left')
    rel_long['valence_label'] = np.where(rel_long['category'] == 'rel_pos', 'positive', 'negative')
    rel_long['delta_from_chance'] = rel_long['prop'] - rel_long['chance']

    rel_long_out = os.path.join(out_dir, f'choice_fixation_relevant_trial_deltas_long_{args.metric}.csv')
    rel_long.to_csv(rel_long_out, index=False)
    print(f'[INFO] Saved relevant trial-level deltas: {rel_long_out} (n_rows={len(rel_long)})')

    # ---- Output 4: Trial-level irrelevant deltas from chance ----
    irr_trials = relsign4_df[['subject', 'game', 'trial_number', 'decision_label', 'irr_pos', 'irr_neg', 'irr_pos_chance', 'irr_neg_chance']].copy()
    irr_long = irr_trials.melt(
        id_vars=['subject', 'game', 'trial_number', 'decision_label'],
        value_vars=['irr_pos', 'irr_neg'],
        var_name='category', value_name='prop')
    irr_chance_long = irr_trials.melt(
        id_vars=['subject', 'game', 'trial_number', 'decision_label'],
        value_vars=['irr_pos_chance', 'irr_neg_chance'],
        var_name='chance_col', value_name='chance')
    irr_chance_long['category'] = irr_chance_long['chance_col'].str.replace('_chance', '', regex=False)
    irr_long = irr_long.merge(
        irr_chance_long[['subject', 'game', 'trial_number', 'decision_label', 'category', 'chance']],
        on=['subject', 'game', 'trial_number', 'decision_label', 'category'], how='left')
    irr_long['valence_label'] = np.where(irr_long['category'] == 'irr_pos', 'positive', 'negative')
    irr_long['delta_from_chance'] = irr_long['prop'] - irr_long['chance']

    irr_long_out = os.path.join(out_dir, f'choice_fixation_irrelevant_trial_deltas_long_{args.metric}.csv')
    irr_long.to_csv(irr_long_out, index=False)
    print(f'[INFO] Saved irrelevant trial-level deltas: {irr_long_out} (n_rows={len(irr_long)})')

    print(f'[INFO] All outputs saved to {out_dir}')


if __name__ == '__main__':
    main()
