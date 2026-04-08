import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import sem
import json
import argparse

# Add argument parser
parser = argparse.ArgumentParser(description='Group-level analysis of recall data')
parser.add_argument('--output_base_dir', type=str, default='../output', help='Base output directory')
parser.add_argument('--data_dir', type=str, default=None, help='Base data directory with per-subject recall outputs (default: output_base_dir)')
parser.add_argument('--roi_type', type=str, default='original', help='ROI type to analyze')
parser.add_argument('--buffer_size', type=int, default=50, help='Buffer size to analyze')
args = parser.parse_args()

output_base_dir = args.output_base_dir
data_dir = args.data_dir if args.data_dir else output_base_dir
roi_type = args.roi_type
buffer_size = args.buffer_size
no_permutation = False

def load_all_subject_data(output_base_dir, roi_type, buffer_size, data_dir=None):
    """
    Load data from all subjects for group analysis.

    Returns:
    - angular_distance_data: Combined angular distance data
    - fixation_summary_data: Combined fixation summary data
    - fixation_time_course_data: Combined time course data
    - subjects: List of successfully loaded subjects
    """
    if data_dir is None:
        data_dir = output_base_dir

    angular_distance_data = []
    fixation_summary_data = []
    fixation_time_course_data = []
    subjects = []

    # Find all subject directories
    subject_dirs = [d for d in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir, d)) and d != 'group']

    # Apply subject-level exclusion based on eyetracking quality (choice trials)
    exclusion_path = os.path.join(output_base_dir, 'choice_trial_drop_overall.csv')
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
        subject_dirs = [d for d in subject_dirs if d not in excluded_subjects]
        print("Excluding subjects from group recall analysis due to eyetracking quality (choice trials):",
              ", ".join(sorted(excluded_subjects)))
    
    for subj in sorted(subject_dirs):
        subj_dir = os.path.join(data_dir, subj)
        
        try:
            # Load angular distance data
            angular_file = f'{subj_dir}/{subj}_summary_free_recall_distance.csv'
            if os.path.exists(angular_file):
                angular_df = pd.read_csv(angular_file)
                angular_df['subject'] = subj
                angular_distance_data.append(angular_df)
            
            # Load fixation summary data
            fixation_file = f'{subj_dir}/{subj}_recall_fixation_analysis_{roi_type}_buffer_{buffer_size}.csv'
            if os.path.exists(fixation_file):
                fixation_df = pd.read_csv(fixation_file)
                fixation_df['subject'] = subj
                fixation_summary_data.append(fixation_df)
            
            # Load time course data
            time_course_file = f'{subj_dir}/{subj}_fixation_time_course_{roi_type}_buffer_{buffer_size}.csv'
            if os.path.exists(time_course_file):
                time_course_df = pd.read_csv(time_course_file)
                time_course_df['subject'] = subj
                fixation_time_course_data.append(time_course_df)
            
            # If we loaded at least one file, add subject to list
            if (os.path.exists(angular_file) or os.path.exists(fixation_file) or 
                os.path.exists(time_course_file)):
                subjects.append(subj)
                
        except Exception as e:
            print(f"Error loading data for subject {subj}: {e}")
            continue
    
    # Combine all data
    angular_combined = pd.concat(angular_distance_data, ignore_index=True) if angular_distance_data else pd.DataFrame()
    fixation_combined = pd.concat(fixation_summary_data, ignore_index=True) if fixation_summary_data else pd.DataFrame()
    time_course_combined = pd.concat(fixation_time_course_data, ignore_index=True) if fixation_time_course_data else pd.DataFrame()
    
    print(f"Loaded data from {len(subjects)} subjects: {subjects}")
    return angular_combined, fixation_combined, time_course_combined, subjects

def cluster_based_permutation_test(
    data_matrix,
    chance_level,
    n_permutations=10000,
    cluster_threshold=0.01,
    cluster_alpha=0.01,
    min_cluster_size=4,
):
    """
    Perform cluster-based permutation test on time series data.
    
    Parameters:
    - data_matrix: Array of shape (n_subjects, n_timepoints) containing subject data
    - chance_level: The null hypothesis value to test against
    - n_permutations: Number of permutation iterations
    - cluster_threshold: p-value threshold for forming clusters
    - cluster_alpha: Alpha level for cluster significance
    
    Returns:
    - significant_timepoints: Boolean array indicating significant time points
    - cluster_pvals: P-values for each cluster
    - observed_clusters: Information about observed clusters
    """
    n_subjects, n_timepoints = data_matrix.shape
    
    # Remove timepoints with too few subjects
    valid_timepoints = np.sum(~np.isnan(data_matrix), axis=0) >= 3
    if not np.any(valid_timepoints):
        return np.zeros(n_timepoints, dtype=bool), [], []
    
    # Step 1: Calculate observed t-statistics for each timepoint
    observed_t_stats = np.full(n_timepoints, np.nan)
    observed_p_values = np.full(n_timepoints, np.nan)
    
    for t in range(n_timepoints):
        if valid_timepoints[t]:
            data_t = data_matrix[:, t]
            valid_data = data_t[~np.isnan(data_t)]
            if len(valid_data) >= 3:
                t_stat, p_val = stats.ttest_1samp(valid_data, chance_level)
                observed_t_stats[t] = t_stat
                observed_p_values[t] = p_val

    # Step 2: Form clusters based on threshold (two-tailed)
    significant_mask = observed_p_values < cluster_threshold
    significant_mask[np.isnan(observed_p_values)] = False
    
    def find_clusters(mask, min_len=1):
        """Find contiguous clusters in boolean mask without bridging; enforce min length."""
        clusters = []
        if not np.any(mask):
            return clusters
        diff = np.diff(np.concatenate(([False], mask, [False])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for start, end in zip(starts, ends):
            cluster_indices = np.arange(start, end)
            if len(cluster_indices) >= min_len:
                # Cluster statistic: sum of |t| over significant timepoints in the cluster
                cluster_sum = np.nansum(np.abs(observed_t_stats[cluster_indices]))
                clusters.append({
                    'indices': cluster_indices,
                    'start': start,
                    'end': end,
                    'size': len(cluster_indices),
                    'sum_t': cluster_sum,
                })
        return clusters
    
    observed_clusters = find_clusters(significant_mask, min_len=min_cluster_size)

    if not observed_clusters:
        print(f"No clusters met initial threshold (p<thr={cluster_threshold}) with min_len={min_cluster_size}; skipping permutations.")
        return np.zeros(n_timepoints, dtype=bool), [], []
    
    # Step 3: Permutation testing
    print(f"Running {n_permutations} permutations for cluster-based test (p<thr={cluster_threshold}, alpha={cluster_alpha}, min_len={min_cluster_size})...")
    
    # Get the maximum cluster statistic from observed data
    max_observed_cluster_stat = max([cluster['sum_t'] for cluster in observed_clusters])
    
    # Permutation distribution
    max_cluster_stats_null = []
    
    for perm in range(n_permutations):
        if perm % 100 == 0:
            print(f"  Permutation {perm}/{n_permutations}")
            
        # Create null data by sign-flipping exactly half of subjects' deviations
        perm_data = data_matrix.copy()
        # Deviation from chance
        deviation = perm_data - chance_level
        # Choose exactly half the subjects to flip (floor if odd)
        n_flip = n_subjects // 2
        flip_subjects = np.random.choice(np.arange(n_subjects), size=n_flip, replace=False)
        flip_mask = np.ones(n_subjects)
        flip_mask[flip_subjects] = -1
        perm_data = chance_level + (flip_mask[:, None] * deviation)
        
        # Calculate permuted t-statistics
        perm_t_stats = np.full(n_timepoints, np.nan)
        perm_p_values = np.full(n_timepoints, np.nan)
        
        for t in range(n_timepoints):
            if valid_timepoints[t]:
                data_t = perm_data[:, t]
                valid_data = data_t[~np.isnan(data_t)]
                if len(valid_data) >= 3:
                    t_stat, p_val = stats.ttest_1samp(valid_data, chance_level)
                    perm_t_stats[t] = t_stat
                    perm_p_values[t] = p_val
        
        # Form clusters for permuted data
        perm_significant_mask = perm_p_values < cluster_threshold
        perm_significant_mask[np.isnan(perm_p_values)] = False
        perm_clusters = find_clusters(perm_significant_mask, min_len=min_cluster_size)

        # Get maximum cluster statistic for this permutation
        if perm_clusters:
            max_perm_cluster_stat = max([cluster['sum_t'] for cluster in perm_clusters])
            max_cluster_stats_null.append(max_perm_cluster_stat)
        else:
            max_cluster_stats_null.append(0)
    
    # Step 4: Calculate cluster p-values
    cluster_pvals = []
    significant_timepoints = np.zeros(n_timepoints, dtype=bool)
    
    for cluster in observed_clusters:
        cluster_stat = cluster['sum_t']
        # P-value is proportion of null distribution >= observed cluster statistic
        p_val = np.mean(np.array(max_cluster_stats_null) >= cluster_stat)
        cluster_pvals.append(p_val)
        
        # Mark timepoints as significant if cluster survives correction
        if p_val < cluster_alpha:
            significant_timepoints[cluster['indices']] = True
    
    print(f"Found {len(observed_clusters)} clusters, {np.sum([p < cluster_alpha for p in cluster_pvals])} significant")
    
    return significant_timepoints, cluster_pvals, observed_clusters

def run_cluster_permutation_test(time_course_data, group_output_dir, roi_type, buffer_size, no_permutation=False):
    """Run cluster-based permutation test and save cluster statistics."""
    if time_course_data.empty:
        print("No time course data to analyze")
        return None, None

    chance_level = 1/6

    # Get unique subjects and time points
    subjects = sorted(time_course_data['subject'].unique())
    time_points = sorted(time_course_data['time_point'].unique())

    # Create matrix for subject data: rows=subjects, cols=time_points (duration only)
    duration_matrix = []

    for subj in subjects:
        subj_data = time_course_data[time_course_data['subject'] == subj]

        subj_duration = []
        for tp in time_points:
            tp_data = subj_data[subj_data['time_point'] == tp]
            if not tp_data.empty:
                subj_duration.append(tp_data['proportion_duration'].iloc[0])
            else:
                subj_duration.append(np.nan)

        duration_matrix.append(subj_duration)

    duration_matrix = np.array(duration_matrix)
    time_points = np.array(time_points)

    # Perform cluster-based permutation test on unsmoothed data (if enabled)
    if no_permutation:
        print("Skipping permutation testing (disabled by --no_permutation flag)")
        significant_timepoints = np.zeros(len(time_points), dtype=bool)
        cluster_pvals = []
        observed_clusters = []
    else:
        print("Performing cluster-based permutation test...")
        perm_n = 1000
        clust_thr = 0.01
        clust_alpha = 0.05
        min_len = 8
        significant_timepoints, cluster_pvals, observed_clusters = cluster_based_permutation_test(
            duration_matrix, chance_level, n_permutations=perm_n, cluster_threshold=clust_thr, cluster_alpha=clust_alpha, min_cluster_size=min_len
        )

    # Compute and save group time course summary (needed by analyze_eyetracking Panel A)
    group_duration_means = np.nanmean(duration_matrix, axis=0)
    group_duration_sems = np.array([sem(duration_matrix[:, i][~np.isnan(duration_matrix[:, i])])
                                   if np.sum(~np.isnan(duration_matrix[:, i])) > 1
                                   else np.nan for i in range(len(time_points))])

    smoothing_window = 10

    def _smooth(y, window_size=3):
        if len(y) < window_size:
            return y
        pad_width = window_size // 2
        padded = np.pad(y, pad_width, mode='edge')
        smoothed = np.convolve(padded, np.ones(window_size) / window_size, mode='valid')
        if len(smoothed) != len(y):
            excess = len(smoothed) - len(y)
            start_trim = excess // 2
            end_trim = excess - start_trim
            smoothed = smoothed[start_trim:-end_trim] if end_trim > 0 else smoothed[start_trim:]
        return smoothed

    smoothed_means = np.full_like(group_duration_means, np.nan)
    smoothed_sems = np.full_like(group_duration_sems, np.nan)
    valid_mask = ~np.isnan(group_duration_means)
    if valid_mask.any() and np.sum(valid_mask) > 3:
        valid_idx = np.where(valid_mask)[0]
        smoothed_means[valid_idx] = _smooth(group_duration_means[valid_mask], window_size=smoothing_window)
        smoothed_sems[valid_idx] = _smooth(group_duration_sems[valid_mask], window_size=smoothing_window)

    group_time_course_df = pd.DataFrame({
        'time_point': time_points,
        'group_duration_mean': group_duration_means,
        'group_duration_sem': group_duration_sems,
        'group_duration_mean_smooth': smoothed_means,
        'group_duration_sem_smooth': smoothed_sems,
        'significant': significant_timepoints,
        'chance_level': chance_level,
        'n_subjects': len(subjects)
    })
    group_time_course_df.to_csv(
        f'{group_output_dir}/group_time_course_{roi_type}_buffer_{buffer_size}.csv', index=False
    )

    # Save cluster statistics
    cluster_stats = {
        'n_clusters': int(len(observed_clusters)),
        'cluster_pvals': [float(p) for p in cluster_pvals],
        'significant_clusters': int(sum([p < (clust_alpha if not no_permutation else 0.05) for p in cluster_pvals])) if not no_permutation else int(sum([p < 0.05 for p in cluster_pvals])),
        'permutation_testing_enabled': not no_permutation,
        'clusters': [
            {
                'start_time': float(time_points[cluster['start']]),
                'end_time': float(time_points[cluster['end']-1]),
                'size': int(cluster['size']),
                'sum_t_statistic': float(cluster['sum_t']),
                'p_value': float(cluster_pvals[i]) if cluster_pvals else None
            }
            for i, cluster in enumerate(observed_clusters)
        ]
    }
    
    json_path = f"{group_output_dir}/group_time_course_clusters_{roi_type}_buffer_{buffer_size}"

    # Save JSON summary
    with open(json_path + ".json", 'w') as f:
        json.dump(cluster_stats, f, indent=2)

    # Also save a human-readable text summary of clusters
    with open(json_path + ".txt", 'w') as f_txt:
        f_txt.write("Group time course cluster summary\n")
        f_txt.write(f"ROI type: {roi_type}\n")
        f_txt.write(f"Buffer size: {buffer_size}\n")
        f_txt.write(f"Permutation testing enabled: {not no_permutation}\n")
        f_txt.write(f"Number of clusters: {len(observed_clusters)}\n")
        if cluster_pvals:
            sig_alpha = clust_alpha if not no_permutation else 0.05
            n_sig = sum([p < sig_alpha for p in cluster_pvals])
            f_txt.write(f"Significant clusters (alpha={sig_alpha}): {n_sig}\n\n")
            for i, (cluster, pval) in enumerate(zip(observed_clusters, cluster_pvals), start=1):
                start_time = float(time_points[cluster['start']])
                end_time = float(time_points[cluster['end']-1])
                size = int(cluster['size'])
                sum_t = float(cluster['sum_t'])
                f_txt.write(
                    f"Cluster {i}: {start_time:.0f} to {end_time:.0f} ms, "
                    f"size={size}, sum_t={sum_t:.3f}, p={pval:.4f}\n"
                )
        else:
            f_txt.write("No clusters or p-values available.\n")
    
    print(f"Cluster statistics saved to {group_output_dir}")
    if no_permutation:
        print("Permutation testing was disabled - no significance testing performed")
    else:
        print(f"Found {len(observed_clusters)} clusters, {sum([p < clust_alpha for p in cluster_pvals])} significant (alpha={clust_alpha})")
        if cluster_pvals:
            for i, (cluster, pval) in enumerate(zip(observed_clusters, cluster_pvals)):
                start_time = time_points[cluster['start']]
                end_time = time_points[cluster['end']-1]
                print(f"  Cluster {i+1}: {start_time:.0f} to {end_time:.0f} ms, p = {pval:.3f}")

def main():
    """Main function to run group analysis."""
    print("="*50)
    print("GROUP LEVEL ANALYSIS")
    print("="*50)
    print(f"ROI type: {roi_type}")
    print(f"Buffer size: {buffer_size}")
    print(f"Permutation testing: {'DISABLED' if no_permutation else 'ENABLED'}")

    # Create output directory
    group_output_dir = os.path.join(output_base_dir, 'eyegaze', 'recall')
    os.makedirs(group_output_dir, exist_ok=True)

    # Load all subject data
    angular_data, fixation_data, time_course_data, subjects = load_all_subject_data(
        output_base_dir, roi_type, buffer_size, data_dir=data_dir
    )

    if not subjects:
        print("No subject data found!")
        return

    if not time_course_data.empty:
        run_cluster_permutation_test(time_course_data, group_output_dir, roi_type, buffer_size, no_permutation)

    print(f"\nGroup analysis complete! Results saved to {group_output_dir}")

if __name__ == "__main__":
    main()
