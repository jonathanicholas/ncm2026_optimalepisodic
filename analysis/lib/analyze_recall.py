import os
import subprocess
import ast
import numpy as np
import math
import json
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pygazeanalyser.edfreader import read_edf
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from PIL import Image
from matplotlib.transforms import Bbox
from matplotlib.image import BboxImage
from scipy.stats import sem
from scipy.stats import pearsonr
from collections import defaultdict
import argparse

'''
Generates:
1. Plots/summaries of angular distance from recalled image from raw x/y coordinates
2. Videos of verbal free recalls and matched eye position over
3. Plots/summarizes proportion of fixations/fixation duration at the encoding location of each item during recall
'''

# Add argument parser

parser = argparse.ArgumentParser(description='Process eyetracking data for a participant')
parser.add_argument('--participant', type=str, required=True, help='Participant ID (e.g., 008)')
parser.add_argument('--file_path', type=str, required=True, help='Path to the analysis files directory')
parser.add_argument('--task_path', type=str, required=True, help='Path to the game info directory')
parser.add_argument('--data_path', type=str, required=True, help='Path to the data logs directory')
parser.add_argument('--img_path', type=str, required=True, help='Path to the stimuli')
parser.add_argument('--make_recall_animations', action='store_true', help='Whether to generate recall animations')
parser.add_argument('--buffer_size', type=int, default=int(os.environ.get('BUFFER_SIZE', 50)), help='Buffer size for ROI processing')
parser.add_argument('--roi_type', type=str, default=os.environ.get('ROI_TYPE', 'original'), help='ROI type: equal_area or original')
args = parser.parse_args()

# get file path
file_path = args.file_path
task_path = args.task_path
data_path = args.data_path
img_path = args.img_path
subj = args.participant

make_recall_animations = args.make_recall_animations # Whether to make animations of eye position during free recall
pre_recall_window = 3000  # Time before recall onset
post_recall_window = 750  # Time after recall onset
buffer_size = args.buffer_size
roi_type = args.roi_type
game_range = [1,8] # min and max number of games played by each participant
circle_radius = 380  # radius of the circle used for generating the stim locations
central_radius = 20 # radius of the central fixation point
# ROI boundary parameters
outer_radius = 530  # base outer boundary of all ROIs
peripheral_buffer = buffer_size  # additional pixels to extend peripheral ROIs outward
remove_overlap = True  # whether to remove overlap between central and peripheral ROIs

####################################
### FUNCTIONS FOR ANGLE ANALYSIS ###
####################################

def calculate_equal_area_central_radius(outer_radius_with_buffer):
    """
    Calculate the central radius that makes all ROIs have equal area.
    """
    return outer_radius_with_buffer / math.sqrt(7)

def is_within_roi_boundary(x_coords, y_coords, center_x, center_y, outer_radius=outer_radius, peripheral_buffer=peripheral_buffer):
    """Return boolean mask for eye positions within outer_radius + peripheral_buffer of display center."""
    actual_outer_radius = outer_radius + peripheral_buffer

    rel_x = x_coords - center_x
    rel_y = y_coords - center_y
    distances = np.sqrt(rel_x**2 + rel_y**2)

    return distances <= actual_outer_radius

def generate_circle_positions(radius=circle_radius, n_positions=6):
    """
    Generate n evenly spaced positions around a circle.
    Starting from top (90 degrees) and going clockwise.
    """
    positions = []
    angles = []
    # Define the order of angles to match the desired positions
    # Starting from top and going clockwise
    angle_order = [90, 30, -30, -90, -150, 150]
    
    for angle_deg in angle_order:
        angle = math.radians(angle_deg)
        x = int(radius * math.cos(angle))
        y = int(radius * math.sin(angle))
        positions.append((x, y))
        angles.append(angle_deg)
    
    return positions, angles

def calculate_angles_to_images(x_coords, y_coords, image_positions, center_x, center_y):
    """
    Calculate the angle between each eye position and each image position.
    
    Parameters:
    - x_coords, y_coords: Arrays of eye tracking coordinates
    - image_positions: List of (x, y) positions of images relative to center
    - center_x, center_y: Center coordinates of the display
    
    Returns:
    - Array of shape (n_samples, n_images) containing angles from each eye position to each image
    """
    # Convert to numpy arrays if they aren't already
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    
    # Calculate absolute positions of images
    abs_image_positions = [(center_x + x, center_y + y) for x, y in image_positions]
    
    # For demonstration, sample every 1000th point (remove for full analysis)
    sample_indices = np.arange(0, len(x_coords))  # Analyze all points
    x_samples = x_coords[sample_indices]
    y_samples = y_coords[sample_indices]
    
    # Initialize results array: shape (n_samples, n_images)
    n_samples = len(sample_indices)
    n_images = len(image_positions)
    angles = np.zeros((n_samples, n_images))
    
    # Calculate angles for each eye position to each image
    for i, (eye_x, eye_y) in enumerate(zip(x_samples, y_samples)):
        # Calculate vector from center to eye position
        eye_vector = [eye_x - center_x, eye_y - center_y]
        eye_magnitude = math.sqrt(eye_vector[0]**2 + eye_vector[1]**2)
        
        for j, (img_x, img_y) in enumerate(abs_image_positions):
            # Calculate vector from center to image
            img_vector = [img_x - center_x, img_y - center_y]
            img_magnitude = math.sqrt(img_vector[0]**2 + img_vector[1]**2)
            
            # Avoid division by zero
            if eye_magnitude == 0 or img_magnitude == 0:
                angles[i, j] = float('nan')
                continue
            
            # Calculate the angle in radians and convert to degrees
            dot_product = eye_vector[0] * img_vector[0] + eye_vector[1] * img_vector[1]
            angle_rad = math.acos(max(min(dot_product / (eye_magnitude * img_magnitude), 1.0), -1.0))
            angle_deg = math.degrees(angle_rad)
            
            # Determine if the angle is clockwise or counterclockwise
            cross_product = eye_vector[0] * img_vector[1] - eye_vector[1] * img_vector[0]
            if cross_product < 0:
                angle_deg = 360 - angle_deg
            
            angles[i, j] = angle_deg
    
    return angles, sample_indices

def process_game_eye_tracking(x_coords, y_coords, curr_positions, curr_recalls, img_names):
    """
    Process eye tracking data for a single game.
    
    Parameters:
    - x_coords, y_coords: Eye tracking coordinates
    - curr_positions: Positions of images
    - curr_recalls: DataFrame of recalls for this game
    - img_names: Names of images in this game
    
    Returns:
    - pre_recall_distances: List of (time, distance) tuples for pre-recall period
    - during_recall_distances: List of (time, distance) tuples for during-recall period
    - eye_data: DataFrame of processed eye tracking data
    """
    pre_recall_distances = []  # Initialize as a list, not a numpy array
    during_recall_distances = []  # Initialize as a list, not a numpy array
    
    center_x = 3840/2
    center_y = 2160/2
    
    # Filter out invalid eye tracking data (0,0 coordinates)
    invalid_coords_mask = (x_coords == 0) & (y_coords == 0)
    
    # Filter out eye positions outside ROI boundaries
    within_roi_mask = is_within_roi_boundary(x_coords, y_coords, center_x, center_y, outer_radius, peripheral_buffer)
    
    # Combine filters: keep positions that are not (0,0) AND are within ROI boundaries
    valid_indices = np.where(~invalid_coords_mask & within_roi_mask)[0]
    
    x_coords_valid = x_coords[valid_indices]
    y_coords_valid = y_coords[valid_indices]
    sample_indices_valid = valid_indices  # Keep track of original indices

    # Calculate angles between eye positions and images
    angles, sample_indices = calculate_angles_to_images(x_coords_valid, y_coords_valid, curr_positions, center_x, center_y)
    
    # Create a dataframe to store eye position data
    eye_data = pd.DataFrame({
        'sample_index': sample_indices_valid
    })
    
    # Add the angle to each image for every eye position
    for i, img_name in enumerate(img_names):
        eye_data[f'angle_to_{img_name}'] = angles[:, i]
    
    # Initialize a column for target image and distance to target
    eye_data['target_image'] = None
    eye_data['distance_to_target'] = np.nan
    eye_data['period_type'] = None  # Add a column to track if it's during recall or between recalls
    
    # Skip the first 1000ms (beep period)
    beep_period_mask = eye_data['sample_index'] < 1000
    eye_data.loc[beep_period_mask, 'period_type'] = 'beep'
    
    # Sort recalls by onset time to process them in order
    sorted_recalls = curr_recalls.sort_values('onset')
    
    # Process each recall period and the period leading up to it
    prev_offset = 1000  # Start after the beep period
    
    # First pass: process the eye data for each recall
    for i, (_, recall) in enumerate(sorted_recalls.iterrows()):
        onset = recall['onset']
        offset = recall['offset']
        recalled_item = recall['item']
        
        # Find the index of the recalled item in img_names
        is_valid_recall = recalled_item in img_names
        
        if is_valid_recall:
            target_idx = img_names.index(recalled_item)
            
            # Assign the period between previous recall and current recall to the upcoming target
            # Only if there's actually a gap between recalls
            if onset > prev_offset:
                between_mask = (eye_data['sample_index'] >= prev_offset) & (eye_data['sample_index'] < onset)
                eye_data.loc[between_mask, 'target_image'] = recalled_item
                eye_data.loc[between_mask, 'period_type'] = 'pre_recall'
                
                # Calculate circular distance to the upcoming target
                between_indices = np.where(between_mask.values)[0]
                if len(between_indices) > 0:
                    target_angles = angles[between_indices, target_idx]
                    circular_distances = np.minimum(np.abs(target_angles), 360 - np.abs(target_angles))
                    eye_data.loc[between_mask, 'distance_to_target'] = circular_distances
            
            # Mark eye positions during this recall period
            recall_mask = (eye_data['sample_index'] >= onset) & (eye_data['sample_index'] <= offset)
            eye_data.loc[recall_mask, 'target_image'] = recalled_item
            eye_data.loc[recall_mask, 'period_type'] = 'during_recall'
            
            # Calculate circular distance to the target image during recall
            recall_indices = np.where(recall_mask.values)[0]
            if len(recall_indices) > 0:
                target_angles = angles[recall_indices, target_idx]
                circular_distances = np.minimum(np.abs(target_angles), 360 - np.abs(target_angles))
                eye_data.loc[recall_mask, 'distance_to_target'] = circular_distances
        
        # Update prev_offset for the next iteration
        prev_offset = offset
    
    # Second pass: collect data for the summary plot
    for i, (_, recall) in enumerate(sorted_recalls.iterrows()):
        onset = recall['onset']
        offset = recall['offset']
        recalled_item = recall['item']
        
        # Only include valid recalls
        if recalled_item in img_names:
            target_idx = img_names.index(recalled_item)
            
            # Get pre-recall window data
            pre_recall_mask = (eye_data['sample_index'] >= onset - pre_recall_window) & (eye_data['sample_index'] < onset)
            
            # Ensure pre-recall window doesn't overlap with other recall periods
            for _, other_recall in sorted_recalls.iterrows():
                other_onset = other_recall['onset']
                other_offset = other_recall['offset']
                # If this is a different recall period that overlaps with our pre-recall window
                if other_onset != onset:  # Not the same recall
                    overlap_mask = (eye_data['sample_index'] >= other_onset) & (eye_data['sample_index'] <= other_offset)
                    # Remove the overlap from our pre-recall mask
                    pre_recall_mask = pre_recall_mask & (~overlap_mask)
            
            pre_recall_indices = np.where(pre_recall_mask.values)[0]
            
            if len(pre_recall_indices) > 0:
                pre_recall_angles = angles[pre_recall_indices, target_idx]
                pre_recall_distances_values = np.minimum(np.abs(pre_recall_angles), 360 - np.abs(pre_recall_angles))
                
                # Normalize time to be relative to recall onset
                pre_recall_times = eye_data.loc[pre_recall_mask, 'sample_index'].values - onset
                
                # Store data for summary plot
                for time, dist in zip(pre_recall_times, pre_recall_distances_values):
                    pre_recall_distances.append((time, dist))            
            # Get during-recall window data (limited to post_recall_window)
            during_recall_mask = (eye_data['sample_index'] >= onset) & (eye_data['sample_index'] < min(onset + post_recall_window, offset))
            during_recall_indices = np.where(during_recall_mask.values)[0]
            
            if len(during_recall_indices) > 0:
                during_recall_angles = angles[during_recall_indices, target_idx]
                during_recall_distances_values = np.minimum(np.abs(during_recall_angles), 360 - np.abs(during_recall_angles))
                
                # Normalize time to be relative to recall onset
                during_recall_times = eye_data.loc[during_recall_mask, 'sample_index'].values - onset
                
                # Store data for summary plot
                for time, dist in zip(during_recall_times, during_recall_distances_values):
                    during_recall_distances.append((time, dist))    
    return pre_recall_distances, during_recall_distances, eye_data

def process_free_recall_data(subj, data_path, task_path, img_path, circle_radius, game_range):
    """
    Process free recall data for a subject and return the collected distance data.
    
    Parameters:
    - subj: Subject ID
    - data_path: Path to data directory
    - task_path: Path to task info directory
    - img_path: Path to image directory
    - circle_radius: Radius of the circle used for generating stimulus locations
    
    Returns:
    - all_pre_recall_distances: List of (time, distance) tuples for pre-recall period
    - all_during_recall_distances: List of (time, distance) tuples for during-recall period
    """
    all_pre_recall_distances = []
    all_during_recall_distances = []
    
    # Read the free recall eyetracking data
    with open(f'../output/{subj}/{subj}_free_recall_eyetracking.pkl', 'rb') as f:
        raw_eyetracking = pickle.load(f)

    # Get the free recall data
    free_recall_df = pd.read_csv(f'{data_path}/{subj}/{subj}_freerecall.csv')
    free_recall_df["onset"] = (free_recall_df["onset"] * 1000).astype(int)
    free_recall_df["offset"] = (free_recall_df["offset"] * 1000).astype(int)

    # Get the positions of each image in each game
    # Load the game info
    with open(f'{task_path}/games_{subj}.json', 'r') as f:
        images_by_game = json.load(f)
    # Load the image positions
    with open(f'{task_path}/positions_{subj}.json', 'r') as f:
        image_positions = json.load(f)

    for curr_game in range(game_range[0], game_range[1]):
        curr_images = images_by_game[curr_game-1]
        pos, _ = generate_circle_positions(radius=circle_radius, n_positions=6)
        curr_positions = [pos[image_positions[img]] for img in curr_images]
        curr_images = [img_path + "/" + img + ".png" for img in curr_images]
        curr_recalls = free_recall_df[free_recall_df["game"] == curr_game]
        img_names = [imgpath.split("/")[-1][:-4] for imgpath in curr_images]

        x_coords = raw_eyetracking[curr_game-1]["x"]
        y_coords = raw_eyetracking[curr_game-1]["y"]

        # Process eye tracking data for this game
        game_pre_recall, game_during_recall, eye_data = process_game_eye_tracking(
            x_coords, y_coords, curr_positions, curr_recalls, img_names
        )
        
        all_pre_recall_distances.extend(game_pre_recall)
        all_during_recall_distances.extend(game_during_recall)
    
    return all_pre_recall_distances, all_during_recall_distances

def create_summary_plot(subj, all_pre_recall_distances, all_during_recall_distances, recall_type, output_dir):
    """
    Create a summary plot of distance to target over time across all games.
    
    Parameters:
    - subj: Subject ID
    - all_pre_recall_distances: List of (time, distance) tuples for pre-recall period
    - all_during_recall_distances: List of (time, distance) tuples for during-recall period
    - plot_dir: Directory to save plots
    - recall_type: Type of recall ('free_recall' or 'value_recall')
    """
    if not all_pre_recall_distances and not all_during_recall_distances:
        return
    
    # Convert to DataFrames for easier processing
    pre_recall_df = pd.DataFrame(all_pre_recall_distances, columns=['time', 'distance'])
    during_recall_df = pd.DataFrame(all_during_recall_distances, columns=['time', 'distance'])
    
    # Combine data
    combined_df = pd.concat([pre_recall_df, during_recall_df])
    
    # Bin the data into small time windows (e.g., 50ms bins)
    bin_size = 50  # ms
    bins = np.arange(-pre_recall_window, post_recall_window + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size/2
    
    # Initialize arrays for mean and SEM
    mean_distances = []
    sem_distances = []
    n_samples = []
    
    # Calculate mean and SEM for each bin
    for i in range(len(bins)-1):
        bin_start = bins[i]
        bin_end = bins[i+1]
        
        bin_data = combined_df[(combined_df['time'] >= bin_start) & (combined_df['time'] < bin_end)]
        
        if not bin_data.empty:
            mean_distances.append(bin_data['distance'].mean())
            sem_distances.append(sem(bin_data['distance']))
            n_samples.append(len(bin_data))
        else:
            mean_distances.append(np.nan)
            sem_distances.append(np.nan)
            n_samples.append(0)
    
    # Chance level for angular distance
    chance_level = 90
    sd_uniform_circle = 103.9
    valid_samples = [n for n in n_samples if n > 0]
    avg_samples_per_bin = np.mean(valid_samples) if valid_samples else 0
    ci_95 = 1.96 * (sd_uniform_circle / np.sqrt(avg_samples_per_bin)) if avg_samples_per_bin > 0 else 0

    # Save the binned data for further analysis
    summary_data = pd.DataFrame({
        'bin_center': bin_centers,
        'mean_distance': mean_distances,
        'sem_distance': sem_distances,
        'n_samples': n_samples,
        'chance_level': chance_level,
        'chance_ci_95': ci_95
    })
        
    os.makedirs(output_dir, exist_ok=True)
    summary_data.to_csv(f'{output_dir}/{subj}_summary_{recall_type}_distance.csv', index=False)

####################################
## FUNCTIONS FOR FIXATION ANALYSIS #
####################################

def load_fixation_dataframe(subj, output_dir, buffer_size, roi_type):
    """
    Load the fixation dataframe with the specified buffer size and ROI type.
    
    Parameters:
    - subj: Subject ID
    - output_dir: Output directory path
    - buffer_size: Buffer size used in ROI processing
    - roi_type: Type of ROI ("equal_area" or "original")
    
    Returns:
    - fixations_df: DataFrame containing fixation data
    """
    if roi_type == "equal_area":
        filename = f'{output_dir}/{subj}_fixations_df_equal_area_buffer_{buffer_size}.csv'
    elif roi_type == "original":
        filename = f'{output_dir}/{subj}_fixations_df_original_buffer_{buffer_size}.csv'
    else:
        raise ValueError(f"Invalid roi_type: {roi_type}. Must be 'equal_area' or 'original'")
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Fixation dataframe not found: {filename}")
    
    return pd.read_csv(filename)

def calculate_fixation_proportions_during_recall(subj, data_path, task_path, output_dir, buffer_size, roi_type, game_range, pre_recall_window, post_recall_window):
    """
    Calculate the proportion of fixations at the encoding location of each recalled item
    using the same time windows as the angular distance analysis.
    
    Parameters:
    - subj: Subject ID
    - data_path: Path to data directory
    - task_path: Path to task info directory
    - output_dir: Output directory path
    - buffer_size: Buffer size used in ROI processing
    - roi_type: Type of ROI ("equal_area" or "original")
    - game_range: Range of games to analyze
    - pre_recall_window: Time before recall onset (ms)
    - post_recall_window: Time after recall onset (ms)
    
    Returns:
    - recall_fixation_analysis: DataFrame with fixation analysis results
    """
    # Load the fixation dataframe
    fixations_df = load_fixation_dataframe(subj, output_dir, buffer_size, roi_type)
    
    # Filter for free recall fixations
    free_recall_fixations = fixations_df[fixations_df["event"] == "free_recall"].copy()
    
    if free_recall_fixations.empty:
        print(f"Warning: No free recall fixations found for subject {subj}")
        return pd.DataFrame()
    
    # Calculate fixation onset/offset relative to recall period start
    free_recall_fixations["fix_onset_relative"] = (
        free_recall_fixations["fix_start"] - free_recall_fixations["eyetracker_onset"]
    )

    # Get the free recall behavioral data
    free_recall_df = pd.read_csv(f'{data_path}/{subj}/{subj}_freerecall.csv')
    free_recall_df["onset"] = (free_recall_df["onset"] * 1000).astype(int)
    free_recall_df["offset"] = (free_recall_df["offset"] * 1000).astype(int)

    # Load game and position information
    with open(f'{task_path}/games_{subj}.json', 'r') as f:
        images_by_game = json.load(f)
    with open(f'{task_path}/positions_{subj}.json', 'r') as f:
        image_positions = json.load(f)
    
    # Initialize results list
    recall_analysis_results = []
    
    # Process each game
    for curr_game in range(game_range[0], game_range[1]):
        curr_images = images_by_game[curr_game-1]
        curr_recalls = free_recall_df[free_recall_df["game"] == curr_game]
        game_fixations = free_recall_fixations[free_recall_fixations["game"] == curr_game]
        
        if curr_recalls.empty or game_fixations.empty:
            continue
        
        # Sort recalls by onset time
        sorted_recalls = curr_recalls.sort_values('onset')
        
        # Process each recall in this game
        for _, recall in sorted_recalls.iterrows():
            onset = recall['onset']
            offset = recall['offset']
            recalled_item = recall['item']
            
            # Check if the recalled item is valid (was in this game)
            if recalled_item not in curr_images:
                continue
            
            # Define time windows matching angular distance analysis
            # Pre-recall window
            pre_recall_mask = (
                (game_fixations['fix_onset_relative'] >= onset - pre_recall_window) & 
                (game_fixations['fix_onset_relative'] < onset)
            )
            
            # Ensure pre-recall window doesn't overlap with other recall periods
            for _, other_recall in sorted_recalls.iterrows():
                other_onset = other_recall['onset']
                other_offset = other_recall['offset']
                if other_onset != onset:  # Not the same recall
                    overlap_mask = (
                        (game_fixations['fix_onset_relative'] >= other_onset) & 
                        (game_fixations['fix_onset_relative'] <= other_offset)
                    )
                    # Remove the overlap from our pre-recall mask
                    pre_recall_mask = pre_recall_mask & (~overlap_mask)
            
            # During-recall window (limited to post_recall_window, same as angular analysis)
            during_recall_mask = (
                (game_fixations['fix_onset_relative'] >= onset) & 
                (game_fixations['fix_onset_relative'] < min(onset + post_recall_window, offset))
            )
            
            # Get fixations for each window
            pre_recall_fixations = game_fixations[pre_recall_mask]
            during_recall_fixations = game_fixations[during_recall_mask]
            
            # Analyze both time windows
            for window_name, window_fixations in [('pre_recall', pre_recall_fixations), 
                                                  ('during_recall', during_recall_fixations)]:
                
                if window_fixations.empty:
                    # No fixations during this window
                    result_row = {
                        'game': curr_game,
                        'recalled_item': recalled_item,
                        'recall_onset': onset,
                        'recall_offset': offset,
                        'recall_duration': offset - onset,
                        'window': window_name,
                        'total_fixations_all': 0,
                        'total_fixations_images': 0,
                        'fixations_at_target': 0,
                        'fixations_at_other_images': 0,
                        'fixations_center': 0,
                        'fixations_none': 0,
                        'proportion_at_target': 0.0,
                        'proportion_at_other_images': 0.0,
                        'total_fixation_duration_all': 0.0,
                        'total_fixation_duration_images': 0.0,
                        'duration_at_target': 0.0,
                        'duration_at_other_images': 0.0,
                        'duration_center': 0.0,
                        'duration_none': 0.0,
                        'proportion_duration_at_target': 0.0,
                        'proportion_duration_at_other_images': 0.0
                    }
                    recall_analysis_results.append(result_row)
                    continue
                
                # Separate fixations by location
                target_fixations = window_fixations[window_fixations['roi_content'] == recalled_item]
                center_fixations = window_fixations[window_fixations['roi_content'] == 'fixation']
                none_fixations = window_fixations[window_fixations['roi_content'] == 'none']
                
                # Other images: valid ROIs that are not target or center
                other_image_fixations = window_fixations[
                    (window_fixations['roi_content'] != recalled_item) & 
                    (window_fixations['roi_content'] != 'fixation') &
                    (window_fixations['roi_content'] != 'none')
                ]
                
                # Image fixations = target + other images (excluding center and none)
                image_fixations = pd.concat([target_fixations, other_image_fixations])
                
                # Count fixations
                total_fixations_all = len(window_fixations)
                total_fixations_images = len(image_fixations)
                fixations_at_target = len(target_fixations)
                fixations_at_other_images = len(other_image_fixations)
                fixations_center = len(center_fixations)
                fixations_none = len(none_fixations)
                
                # Calculate fixation durations
                total_duration_all = window_fixations['fix_duration_bounded'].sum()
                total_duration_images = image_fixations['fix_duration_bounded'].sum()
                duration_at_target = target_fixations['fix_duration_bounded'].sum()
                duration_at_other_images = other_image_fixations['fix_duration_bounded'].sum()
                duration_center = center_fixations['fix_duration_bounded'].sum()
                duration_none = none_fixations['fix_duration_bounded'].sum()
                
                # Calculate proportions (excluding center - denominator = target + other images only)
                if total_fixations_images > 0:
                    prop_at_target = fixations_at_target / total_fixations_images
                    prop_at_other_images = fixations_at_other_images / total_fixations_images
                else:
                    prop_at_target = 0.0
                    prop_at_other_images = 0.0
                
                # Calculate duration proportions (excluding center)
                if total_duration_images > 0:
                    prop_duration_at_target = duration_at_target / total_duration_images
                    prop_duration_at_other_images = duration_at_other_images / total_duration_images
                else:
                    prop_duration_at_target = 0.0
                    prop_duration_at_other_images = 0.0
                
                recall_analysis_results.append({
                    'game': curr_game,
                    'recalled_item': recalled_item,
                    'recall_onset': onset,
                    'recall_offset': offset,
                    'recall_duration': offset - onset,
                    'window': window_name,
                    'total_fixations_all': total_fixations_all,
                    'total_fixations_images': total_fixations_images,
                    'fixations_at_target': fixations_at_target,
                    'fixations_at_other_images': fixations_at_other_images,
                    'fixations_center': fixations_center,
                    'fixations_none': fixations_none,
                    'proportion_at_target': prop_at_target,
                    'proportion_at_other_images': prop_at_other_images,
                    'total_fixation_duration_all': total_duration_all,
                    'total_fixation_duration_images': total_duration_images,
                    'duration_at_target': duration_at_target,
                    'duration_at_other_images': duration_at_other_images,
                    'duration_center': duration_center,
                    'duration_none': duration_none,
                    'proportion_duration_at_target': prop_duration_at_target,
                    'proportion_duration_at_other_images': prop_duration_at_other_images
                })
    
    return pd.DataFrame(recall_analysis_results)

def create_fixation_proportion_summary(subj, recall_fixation_analysis, output_dir, buffer_size, roi_type):
    """
    Create summary plots and statistics for fixation proportions during recall.
    Now simplified to exclude center ROI from proportion calculations and show only duration proportions.
    """
    if recall_fixation_analysis.empty:
        print(f"No recall fixation data to analyze for subject {subj}")
        return
    
    # Calculate summary statistics for each window
    windows = ['pre_recall', 'during_recall']
    summary_stats = {}
    
    for window in windows:
        window_data = recall_fixation_analysis[recall_fixation_analysis['window'] == window]
        if not window_data.empty:
            summary_stats[window] = {
                'mean_proportion_duration_at_target': window_data['proportion_duration_at_target'].mean(),
                'total_recalls_analyzed': len(window_data),
                'recalls_with_target_fixations': len(window_data[window_data['fixations_at_target'] > 0]),
                'mean_image_fixations_per_recall': window_data['total_fixations_images'].mean(),
            }
    
    # Save summary statistics
    summary_filename = f'{output_dir}/{subj}_recall_fixation_summary_{roi_type}_buffer_{buffer_size}.json'
    with open(summary_filename, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Save detailed analysis
    analysis_filename = f'{output_dir}/{subj}_recall_fixation_analysis_{roi_type}_buffer_{buffer_size}.csv'
    recall_fixation_analysis.to_csv(analysis_filename, index=False)
    
    print(f"Fixation proportion analysis completed for subject {subj}")
    for window in windows:
        if window in summary_stats:
            print(f"{window.replace('_', ' ').title()} window:")
            print(f"  Mean duration proportion at target: {summary_stats[window]['mean_proportion_duration_at_target']:.3f} (chance = {chance_level:.3f})")
    print(f"Saved summary: {summary_filename}")
    print(f"Saved analysis: {analysis_filename}")

def calculate_fixation_time_course(subj, data_path, task_path, output_dir, buffer_size, roi_type, game_range, pre_recall_window, post_recall_window):
    """
    Calculate fixation proportions over time with a running average approach.
    Simplified to exclude center ROI from proportion calculations.
    """
    # Load the fixation dataframe
    fixations_df = load_fixation_dataframe(subj, output_dir, buffer_size, roi_type)
    
    # Filter for free recall fixations
    free_recall_fixations = fixations_df[fixations_df["event"] == "free_recall"].copy()
    
    if free_recall_fixations.empty:
        print(f"Warning: No free recall fixations found for subject {subj}")
        return pd.DataFrame()
    
    # Calculate fixation onset/offset relative to recall period start
    free_recall_fixations["fix_onset_relative"] = (
        free_recall_fixations["fix_start"] - free_recall_fixations["eyetracker_onset"]
    )
    
    # Get the free recall behavioral data
    free_recall_df = pd.read_csv(f'{data_path}/{subj}/{subj}_freerecall.csv')
    free_recall_df["onset"] = (free_recall_df["onset"] * 1000).astype(int)
    free_recall_df["offset"] = (free_recall_df["offset"] * 1000).astype(int)

    # Load game and position information
    with open(f'{task_path}/games_{subj}.json', 'r') as f:
        images_by_game = json.load(f)
    with open(f'{task_path}/positions_{subj}.json', 'r') as f:
        image_positions = json.load(f)

    # Collect all fixations with their timing relative to recall onset
    all_fixation_data = []
    
    # Process each game
    for curr_game in range(game_range[0], game_range[1]):
        curr_images = images_by_game[curr_game-1]
        curr_recalls = free_recall_df[free_recall_df["game"] == curr_game]
        game_fixations = free_recall_fixations[free_recall_fixations["game"] == curr_game]
        
        if curr_recalls.empty or game_fixations.empty:
            continue
        
        # Sort recalls by onset time
        sorted_recalls = curr_recalls.sort_values('onset')
        
        # Process each recall in this game
        for _, recall in sorted_recalls.iterrows():
            onset = recall['onset']
            offset = recall['offset']
            recalled_item = recall['item']
            
            # Check if the recalled item is valid (was in this game)
            if recalled_item not in curr_images:
                continue
            
            # Get all fixations in the extended window (pre + during)
            extended_window_mask = (
                (game_fixations['fix_onset_relative'] >= onset - pre_recall_window) & 
                (game_fixations['fix_onset_relative'] <= min(onset + post_recall_window, offset))
            )
            
            # Apply overlap exclusion for pre-recall period only
            for _, other_recall in sorted_recalls.iterrows():
                other_onset = other_recall['onset']
                other_offset = other_recall['offset']
                if other_onset != onset:  # Not the same recall
                    # Only exclude overlaps in the pre-recall period
                    overlap_in_pre_recall = (
                        (game_fixations['fix_onset_relative'] >= onset - pre_recall_window) & 
                        (game_fixations['fix_onset_relative'] < onset) &
                        (game_fixations['fix_onset_relative'] >= other_onset) & 
                        (game_fixations['fix_onset_relative'] <= other_offset)
                    )
                    extended_window_mask = extended_window_mask & (~overlap_in_pre_recall)
            
            window_fixations = game_fixations[extended_window_mask].copy()
            
            if window_fixations.empty:
                continue
            
            # Calculate time relative to recall onset for each fixation
            window_fixations['time_relative_to_onset'] = window_fixations['fix_onset_relative'] - onset
            
            # Add metadata for each fixation
            for _, fix in window_fixations.iterrows():
                # Determine if fixation is at target
                is_target = (fix['roi_content'] == recalled_item) if pd.notna(fix['roi_content']) else False
                is_center = (fix['roi_content'] == 'fixation') if pd.notna(fix['roi_content']) else False
                is_other_image = (
                    pd.notna(fix['roi_content']) and 
                    fix['roi_content'] != 'none' and 
                    fix['roi_content'] != 'fixation' and 
                    fix['roi_content'] != recalled_item
                )
                is_image = is_target or is_other_image  # Any image ROI (excluding center and none)
                
                all_fixation_data.append({
                    'game': curr_game,
                    'recalled_item': recalled_item,
                    'time_relative_to_onset': fix['time_relative_to_onset'],
                    'fixation_duration': fix['fix_duration_bounded'],
                    'is_target': is_target,
                    'is_center': is_center,
                    'is_other_image': is_other_image,
                    'is_image': is_image,
                    'roi_content': fix['roi_content']
                })
    
    return pd.DataFrame(all_fixation_data)

def create_time_course_plot(subj, time_course_data, output_dir, buffer_size, roi_type, pre_recall_window, post_recall_window):
    """
    Create time course plots showing fixation proportions using a sliding window.
    No smoothing, proportions only, plotted as lines.
    """
    if time_course_data.empty:
        print(f"No time course data to plot for subject {subj}")
        return
    
    # Parameters for sliding window
    window_size = 100  # Size of sliding window in ms, was 50
    step_size = 25     # Step size for sliding window in ms, was 25
    
    # Define time range
    time_min = -pre_recall_window
    time_max = post_recall_window
    
    # Create time points for sliding window centers
    time_points = np.arange(time_min + window_size/2, time_max - window_size/2 + step_size, step_size)
    
    # Initialize arrays for proportions
    proportions_count = []
    proportions_duration = []
    n_fixations_per_window = []
    n_image_fixations_per_window = []
    
    # Calculate proportions for each sliding window
    for center_time in time_points:
        window_start = center_time - window_size/2
        window_end = center_time + window_size/2
        
        # Get fixations in this sliding window
        window_data = time_course_data[
            (time_course_data['time_relative_to_onset'] >= window_start) & 
            (time_course_data['time_relative_to_onset'] < window_end)
        ]
        
        if len(window_data) == 0:
            proportions_count.append(np.nan)
            proportions_duration.append(np.nan)
            n_fixations_per_window.append(0)
            n_image_fixations_per_window.append(0)
            continue
        
        # Filter for image fixations only (exclude center and none)
        image_data = window_data[window_data['is_image']]
        
        if len(image_data) == 0:
            proportions_count.append(np.nan)
            proportions_duration.append(np.nan)
            n_fixations_per_window.append(len(window_data))
            n_image_fixations_per_window.append(0)
            continue
        
        # Count fixations
        n_target = image_data['is_target'].sum()
        n_image_total = len(image_data)
        
        # Calculate fixation count proportion
        prop_count = n_target / n_image_total if n_image_total > 0 else 0
        
        # Calculate duration proportion
        duration_target = image_data[image_data['is_target']]['fixation_duration'].sum()
        duration_total = image_data['fixation_duration'].sum()
        prop_duration = duration_target / duration_total if duration_total > 0 else 0
        
        proportions_count.append(prop_count)
        proportions_duration.append(prop_duration)
        n_fixations_per_window.append(len(window_data))
        n_image_fixations_per_window.append(n_image_total)
    
    # Convert to numpy arrays
    proportions_count = np.array(proportions_count)
    proportions_duration = np.array(proportions_duration)
    
    # Define chance level
    chance_level = 1/6

    # Save the time course data
    time_course_summary = pd.DataFrame({
        'time_point': time_points,
        'proportion_count': proportions_count,
        'proportion_duration': proportions_duration,
        'n_fixations_per_window': n_fixations_per_window,
        'n_image_fixations_per_window': n_image_fixations_per_window,
        'window_size_ms': window_size,
        'step_size_ms': step_size,
        'chance_level': chance_level
    })
    
    time_course_filename = f'{output_dir}/{subj}_fixation_time_course_{roi_type}_buffer_{buffer_size}.csv'
    time_course_summary.to_csv(time_course_filename, index=False)
    
    print(f"Time course analysis completed for subject {subj}")
    print(f"Used sliding window: {window_size}ms window, {step_size}ms steps")
    print(f"Chance level: {chance_level:.3f} (1/6 images)")
    print(f"Saved time course data: {time_course_filename}")
    
    return time_course_summary

####################################
### FUNCTIONS FOR ANIMATIONS #######
####################################

def draw_roi_overlay(ax, center_x, center_y, radius=530, central_radius=central_radius, video=False):
    """
    Draw the ROI overlay on the given axes.
    
    Parameters:
    - ax: matplotlib axes
    - center_x, center_y: center coordinates
    - radius: radius of the outer circle
    - central_radius: radius of the central circle
    """
    # Draw the outer circle
    circle = plt.Circle((center_x, center_y), radius, 
                       fill=False, color='black', 
                       linestyle='-', linewidth=1)
    ax.add_patch(circle)
    
    # Draw the central circle
    central_circle = plt.Circle((center_x, center_y), central_radius, 
                              fill=False, color='black', 
                              linestyle='-', linewidth=1)
    ax.add_patch(central_circle)
    
    # Get screen dimensions
    screen_width = 3840
    screen_height = 2160
    
    # Calculate the maximum distance from center to any corner of the screen
    max_distance = max(
        np.sqrt((0 - center_x)**2 + (0 - center_y)**2),
        np.sqrt((screen_width - center_x)**2 + (0 - center_y)**2),
        np.sqrt((0 - center_x)**2 + (screen_height - center_y)**2),
        np.sqrt((screen_width - center_x)**2 + (screen_height - center_y)**2)
    )
    
    # Draw the sector lines every 60 degrees, but rotated by -30 degrees
    for i in range(6):
        angle = i * 60 - 30  # Rotate by -30 degrees
        rad = math.radians(90 - angle)  # Convert back to math angles
        
        # Calculate start point on central circle
        start_x = center_x + central_radius * math.cos(rad)
        start_y = center_y + central_radius * math.sin(rad)
                
        # Always draw the original lines to the circle
        end_x_circle = center_x + radius * math.cos(rad)
        end_y_circle = center_y + radius * math.sin(rad)
        
        # Draw line from central circle to outer circle
        line = plt.Line2D([start_x, end_x_circle], [start_y, end_y_circle], 
                         color='black', linestyle='-', linewidth=1)
        ax.add_line(line)

    if not video:
        # Draw white squares at stimulus positions
        positions, _ = generate_circle_positions()
        square_size = 160  # 160x160 pixel squares
        half_size = square_size / 2
        
        for x, y in positions:
            # Calculate absolute position
            abs_x = center_x + x
            abs_y = center_y + y
            
            # Create a white square centered at the stimulus position
            square = plt.Rectangle((abs_x - half_size, abs_y - half_size), 
                                square_size, square_size,
                                fill=False, edgecolor='white', linewidth=2)
            ax.add_patch(square)

def add_images_to_plot(ax, images, positions, center_x, center_y, size=160, alpha=0.7):
    """
    Add images to the plot at specified positions with a fixed size.
    
    Parameters:
    - ax: matplotlib axis
    - images: list of image paths
    - positions: list of (x, y) positions relative to center
    - center_x, center_y: center coordinates
    - size: size of images in pixels (width and height)
    - alpha: transparency (lower value = more greyed out)
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    
    # Calculate half size for positioning
    half_size = size / 2
    
    for img_path, (rel_x, rel_y) in zip(images, positions):
        try:
            # Calculate absolute position
            abs_x = center_x + rel_x
            abs_y = center_y + rel_y
            
            # Load and resize the image using PIL
            pil_img = Image.open(img_path)
            pil_img = pil_img.resize((size, size), Image.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(pil_img)
            
            # Calculate the extent for imshow (left, right, bottom, top)
            extent = [
                abs_x - half_size, 
                abs_x + half_size, 
                abs_y + half_size,  # Note: y-axis is inverted in matplotlib
                abs_y - half_size
            ]
            
            # Display the image using imshow with exact positioning
            ax.imshow(img_array, extent=extent, alpha=alpha, zorder=0)
            
        except Exception as e:
            print(f"Error adding image {img_path}: {e}")

def create_animation_with_audio(subj, game_num, x_coords, y_coords, images, positions, 
                               center_x, center_y, phase):
    """
    Create an animation of eye-tracking data with synchronized audio.
    
    Parameters:
    - subj: subject number
    - game_num: game number
    - x_coords, y_coords: eye-tracking coordinates (assumed to be 1000 Hz)
    - images: list of image paths
    - positions: list of image positions
    - center_x, center_y: center coordinates
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")  # Use Agg backend for saving without display
    from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip
    import os
    import tempfile
    from tqdm import tqdm
    import wave
    print(f"Creating recall animation for subject {subj}, game {game_num}...")
    
    # Create the output file name

    videos_dir = f"../output/{subj}/videos"
    os.makedirs(videos_dir, exist_ok=True)
    output_filename = f"{videos_dir}/{subj}_{phase}_game_{game_num}.mp4"
    
    # Audio file path
    audio_file = f"{data_path}/{subj}/freerecall/{subj}_{phase}_{game_num-1}.mp3"
    
    # Get audio duration
    from pydub import AudioSegment
    audio = AudioSegment.from_mp3(audio_file)
    audio_duration = len(audio) / 1000.0  # Duration in seconds
    # Eye-tracking sampling rate (Hz)
    eyetracking_rate = 1000  # Assuming 1000 Hz
    
    # Total frames in eye-tracking data
    total_frames = len(x_coords)
    
    # Calculate actual eye-tracking duration
    eyetracking_duration = total_frames / eyetracking_rate
    
    print(f"Audio duration: {audio_duration:.2f} seconds")
    print(f"Eye-tracking duration: {eyetracking_duration:.2f} seconds")
    
    # Determine if we need to adjust durations
    if abs(eyetracking_duration - audio_duration) > 0.1:
        print(f"WARNING: Duration mismatch between eye-tracking and audio")
        # Use the shorter of the two durations
        duration = min(eyetracking_duration, audio_duration)
    else:
        duration = audio_duration
    
    # Target FPS for the output video
    target_fps = 30
    
    # Create time points for each video frame, evenly spaced across the duration
    video_frame_count = int(duration * target_fps)
    time_points = np.linspace(0, duration, video_frame_count, endpoint=False)
    
    # Convert time points to eye-tracking frame indices
    # This ensures consistent sampling throughout the video
    frame_indices = [min(int(t * eyetracking_rate), total_frames-1) for t in time_points]
    
    print(f"Rendering {len(frame_indices)} video frames at {target_fps} fps")
    print(f"Video duration will be: {duration:.2f} seconds")
    
    # Create a directory for temporary frame images
    temp_dir = tempfile.mkdtemp()
    frame_files = []
    
    # Trail length in seconds
    trail_duration = 0.5  # 0.5 seconds
    trail_frames = int(trail_duration * eyetracking_rate)
    
    # Render each frame
    for i, frame_idx in enumerate(tqdm(frame_indices)):
        # Create a new figure for each frame
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100, facecolor='white')
        # ax.set_xlim(0, 3840)
        # ax.set_ylim(0, 2160)
        ax.set_xlim(1000, 2800)
        ax.set_ylim(500, 1650)
        ax.invert_yaxis()
        ax.set_facecolor('gray')
        
        # Remove ticks and ticklabels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Draw static elements
        draw_roi_overlay(ax, center_x, center_y, video=True)
        add_images_to_plot(ax, images, positions, center_x, center_y, size=160, alpha=0.5)
        
        # Get current position
        current_x = x_coords[frame_idx]
        current_y = y_coords[frame_idx]
        
        # Plot current position
        ax.scatter(current_x, current_y, s=80, color='blue', alpha=0.8)
        
        # Plot trail
        # start_idx = max(0, frame_idx - trail_frames)
        # trail_x = x_coords[start_idx:frame_idx+1]
        # trail_y = y_coords[start_idx:frame_idx+1]
        # ax.plot(trail_x, trail_y, 'b-', alpha=0.3, linewidth=1)
        
        # Update title with time in seconds
        time_in_seconds = time_points[i]
        ax.set_title(f'Time = {time_in_seconds:.2f} s')
        
        # Save frame
        frame_file = os.path.join(temp_dir, f"frame_{i:04d}.png")
        fig.savefig(frame_file, dpi=100)
        plt.close(fig)
        frame_files.append(frame_file)
    
    print("Creating video from frames...")
    
    # Create video from frames
    clip = ImageSequenceClip(frame_files, fps=target_fps)
    
    # Add audio
    audio = AudioFileClip(audio_file)
    if audio.duration > duration:
        audio = audio.subclip(0, duration)
    
    # Set the audio of the video
    clip = clip.set_audio(audio)
    
    # Write the final video with audio
    clip.write_videofile(output_filename, codec='libx264', audio_codec='aac', 
                        threads=4, fps=target_fps)
    
    # Clean up temporary files
    for file in frame_files:
        os.unlink(file)
    os.rmdir(temp_dir)
    
    print(f"Animation with audio saved as {output_filename}")
    return output_filename

def generate_freerecall_animations(subj):
    # Read the free recall data
    with open(f'../output/{subj}/{subj}_free_recall_eyetracking.pkl', 'rb') as f:
        raw_eyetracking = pickle.load(f)

    # Get the positions of each image in each game
    # Load the game info
    with open(f'{task_path}/games_{subj}.json', 'r') as f:
        images_by_game = json.load(f)
    # Load the image positions
    with open(f'{task_path}/positions_{subj}.json', 'r') as f:
        image_positions = json.load(f)

    for curr_game in range(1, 9):
        curr_images = images_by_game[curr_game-1]
        pos, _ = generate_circle_positions(radius=circle_radius, n_positions=6)
        curr_positions = [pos[image_positions[img]] for img in curr_images]
        curr_images = [img_path + "/" + img + ".png" for img in curr_images]

        x_coords = raw_eyetracking[curr_game-1]["x"]
        y_coords = raw_eyetracking[curr_game-1]["y"]

        center_x = 3840/2
        center_y = 2160/2

        create_animation_with_audio(subj, curr_game, x_coords, y_coords, curr_images, curr_positions, center_x, center_y, "freerecall")

print("\n" + "="*50)
print("ANGULAR DISTANCE ANALYSIS")
print("="*50)

# Angular distance analysis
output_dir = os.path.join(data_path, subj)
all_pre_recall_distances, all_during_recall_distances = process_free_recall_data(subj, data_path, task_path, img_path, circle_radius, game_range)
create_summary_plot(
    subj, all_pre_recall_distances, all_during_recall_distances,
    'free_recall', output_dir
)

print("\n" + "="*50)
print("FIXATION PROPORTION ANALYSIS")
print("="*50)

try:
    recall_fixation_analysis = calculate_fixation_proportions_during_recall(
        subj, data_path, task_path, output_dir, buffer_size, roi_type, game_range, pre_recall_window, post_recall_window
    )
    
    create_fixation_proportion_summary(
        subj, recall_fixation_analysis, output_dir, buffer_size, roi_type
    )
    
    time_course_data = calculate_fixation_time_course(
        subj, data_path, task_path, output_dir, buffer_size, roi_type, game_range, pre_recall_window, post_recall_window
    )
    if not time_course_data.empty:
        time_course_summary = create_time_course_plot(
            subj, time_course_data, output_dir, buffer_size, roi_type, pre_recall_window, post_recall_window
        )
    
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Make sure you have run process_eyetracking.py first to generate the fixation dataframes.")
except Exception as e:
    print(f"Error in fixation proportion analysis: {e}")

# Create animation of free recall eye position
if make_recall_animations:
    print("\n" + "="*50)
    print("CREATING RECALL ANIMATIONS")
    print("="*50)
    generate_freerecall_animations(subj)
