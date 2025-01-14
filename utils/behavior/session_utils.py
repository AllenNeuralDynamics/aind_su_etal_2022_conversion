import numpy as np
import os
import scipy.stats as stats
from collections import defaultdict
from typing import List, Dict, Any
from sklearn.linear_model import LogisticRegression

import os
import re
import numpy as np
import pandas as pd
from scipy.stats import zscore
from utils.basics.data_org import curr_computer, parse_session_string
import matplotlib.pyplot as plt
from scipy.io import loadmat
from itertools import chain

def clean_up_licks(licksL, licksR, crosstalk_thresh=100, rebound_thresh=50, plot=False):
    """
    Clean up lick times by removing elements based on crosstalk and rebound thresholds.
    
    Parameters:
    licksL (list or np.ndarray): Vector of lick times for the left side (in ms).
    licksR (list or np.ndarray): Vector of lick times for the right side (in ms).
    crosstalk_thresh (float): Time threshold (in ms) for detecting crosstalk.
    rebound_thresh (float): Time threshold (in ms) for rebound filtering.
    plot (bool): Whether to plot histograms before and after clean-up.
    
    Returns:
    tuple: (licksL_cleaned, licksR_cleaned), cleaned vectors of lick times for left and right.
    """
    # Sort inputs to ensure time order
    licksL = np.sort(licksL)
    licksR = np.sort(licksR)

    # Crosstalk filtering
    licksL_cleaned = licksL[
        ~np.array([np.any((licksR < x) & ((x - licksR) <= crosstalk_thresh)) for x in licksL])
    ]
    licksR_cleaned = licksR[
        ~np.array([np.any((licksL < x) & ((x - licksL) <= crosstalk_thresh)) for x in licksR])
    ]

    # Rebound filtering
    licksL_cleaned = licksL_cleaned[np.insert(np.diff(licksL_cleaned) > rebound_thresh, 0, True)]
    licksR_cleaned = licksR_cleaned[np.insert(np.diff(licksR_cleaned) > rebound_thresh, 0, True)]

    # Plot results if requested
    if plot:
        bins_same = np.linspace(0, 300, 30)
        bins_diff = np.linspace(0, 300, 30)

        def plot_histogram(licks, title, ylabel):
            plt.hist(licks, bins=bins_same if "ILI" in title else bins_diff, edgecolor="none")
            plt.title(title)
            if ylabel:
                plt.ylabel(ylabel)

        # Before clean-up
        all_licks = np.concatenate([licksL, licksR])
        all_licks_id = np.concatenate([np.zeros_like(licksL), np.ones_like(licksR)])
        sorted_indices = np.argsort(all_licks)
        all_licks = all_licks[sorted_indices]
        all_licks_id = all_licks_id[sorted_indices]
        all_licks_diff = np.diff(all_licks)
        id_pre = all_licks_id[:-1]
        id_post = all_licks_id[1:]

        fig = plt.figure(figsize=(12, 6))
        plt.subplot(2, 4, 1)
        plot_histogram(all_licks_diff[(id_pre == 0) & (id_post == 0)], 'L_ILI', 'Before clean-up')
        plt.subplot(2, 4, 2)
        plot_histogram(all_licks_diff[(id_pre == 1) & (id_post == 1)], 'R_ILI', None)
        plt.subplot(2, 4, 3)
        plot_histogram(all_licks_diff[(id_pre == 0) & (id_post == 1)], 'L-R_ILI', None)
        plt.subplot(2, 4, 4)
        plot_histogram(all_licks_diff[(id_pre == 1) & (id_post == 0)], 'R-L_ILI', None)

        # After clean-up
        all_licks = np.concatenate([licksL_cleaned, licksR_cleaned])
        all_licks_id = np.concatenate([np.zeros_like(licksL_cleaned), np.ones_like(licksR_cleaned)])
        sorted_indices = np.argsort(all_licks)
        all_licks = all_licks[sorted_indices]
        all_licks_id = all_licks_id[sorted_indices]
        all_licks_diff = np.diff(all_licks)
        id_pre = all_licks_id[:-1]
        id_post = all_licks_id[1:]

        plt.subplot(2, 4, 5)
        plot_histogram(all_licks_diff[(id_pre == 0) & (id_post == 0)], 'L_ILI', 'After clean-up')
        plt.subplot(2, 4, 6)
        plot_histogram(all_licks_diff[(id_pre == 1) & (id_post == 1)], 'R_ILI', None)
        plt.subplot(2, 4, 7)
        plot_histogram(all_licks_diff[(id_pre == 0) & (id_post == 1)], 'L-R_ILI', None)
        plt.subplot(2, 4, 8)
        plot_histogram(all_licks_diff[(id_pre == 1) & (id_post == 0)], 'R-L_ILI', None)

        plt.tight_layout()
        plt.show()
    else:
        fig = None

    return licksL_cleaned, licksR_cleaned, fig


def beh_analysis_no_plot_opmd(session_name, rev_for_flag=0, make_fig_flag=0, t_max=10, 
                              time_max=61000, time_bins=6, simple_flag=0):
    # Parse input parameters
    bin_size = (time_max - 1000) / time_bins
    time_bin_edges = np.arange(1000, time_max + 1, bin_size)
    t_max = len(time_bin_edges) - 1

    # Simulate currComputer() function
    root = "/path/to/data"
    sep = "/"

    # Extract animal name and date
    animal_name, date = session_name.split('d', 1)
    animal_name = animal_name[1:]
    date = date[:9]
    session_folder = f"m{animal_name}{date}"

    # Determine session data path
    if session_name[-1].isalpha():
        session_data_path = f"{root}{sep}{animal_name}{sep}{session_folder}{sep}sorted{sep}session {session_name[-1]}{sep}{session_name}_sessionData_behav.mat"
    else:
        session_data_path = f"{root}{sep}{animal_name}{sep}{session_folder}{sep}sorted{sep}session{sep}{session_name}_sessionData_behav.mat"

    # Load or generate session data
    beh_session_data = None
    if rev_for_flag:
        try:
            # Simulate loading MAT file
            # beh_session_data = loadmat(session_data_path)['behSessionData']
            pass
        except FileNotFoundError:
            beh_session_data, block_switch, block_probs = generate_session_data_behav_operant_matching(session_name)
    else:
        try:
            # Simulate loading MAT file
            # beh_session_data = loadmat(session_data_path)['behSessionData']
            pass
        except FileNotFoundError:
            beh_session_data, block_switch_l, block_switch_r = generate_session_data_operant_matching_decoupled_rwd_delay(session_name)

    # Process behavioral session data
    response_inds = [i for i, trial in enumerate(beh_session_data) if not np.isnan(trial['rewardTime'])]
    omit_inds = [i for i, trial in enumerate(beh_session_data) if np.isnan(trial['rewardTime'])]

    rwd_delay = np.mean([trial['rewardTime'] - trial['respondTime'] for trial in beh_session_data if trial['rewardTime']])
    all_reward_r = np.array([trial.get('rewardR', 0) for trial in beh_session_data])
    all_reward_l = np.array([trial.get('rewardL', 0) for trial in beh_session_data])
    all_choices = np.nan * np.ones(len(beh_session_data))
    all_choices[~np.isnan(all_reward_r)] = 1
    all_choices[~np.isnan(all_reward_l)] = -1

    # Additional processing for lick latencies, rewards, no rewards, etc.
    lick_lat = [trial['respondTime'] - trial['CSon'] for trial in beh_session_data if not np.isnan(trial['rewardTime'])]
    lick_rate = calculate_lick_rate(beh_session_data, rwd_delay)
    lick_rate_z = zscore(lick_rate)

    # Prepare output structure
    results = {
        'allChoices': all_choices,
        'allRewards': all_reward_r - all_reward_l,
        'lickLat': lick_lat,
        'lickRate': lick_rate,
        'lickRateZ': lick_rate_z,
        'rwdDelay': rwd_delay,
        # Add more fields as necessary
    }

    if simple_flag:
        return results

    # Perform additional analyses (GLMs, regressions, etc.)
    # ...

    return results


def load_df_from_mat(file_path):
    # initialization
    beh_df = pd.DataFrame()
    licks_L = []
    licks_R = []
    # Load the .mat file
    mat_data = loadmat(file_path)

    # Access the 'beh' struct
    beh = mat_data['behSessionData']

    # Convert the struct fields to a dictionary
    # Assuming 'beh' is a MATLAB struct with fields accessible via numpy record arrays
    if isinstance(beh, np.ndarray) and beh.dtype.names is not None:
        beh_dict = {field: beh[field].squeeze() for field in beh.dtype.names}

        # Create a DataFrame from the dictionary
        beh_df = pd.DataFrame(beh_dict)
        beh_df.head(10)
        # for column in beh_df.columns:
        #     if beh_df[column].dtype == np.object:
        #         beh_df[column] = beh_df[column].str[0]
        for column in beh_df.columns:
            if column in ['trialEnd', 'CSon', 'respondTime', 'rewardTime', 'rewardProbL', 'rewardProbR', 'laser', 'rewardL', 'rewardR']:
                curr_list = beh_df[column].tolist()
                curr_list = [x[0][0] for x in curr_list]
                beh_df[column] = curr_list
            elif column in ['licksL', 'licksR', 'trialType']:
                curr_list = beh_df[column].tolist()
                curr_list = [x[0] if len(x)>0 else x for x in curr_list] 
                beh_df[column] = curr_list
        # all licks
        licks_L = list(chain.from_iterable(beh_df['licksL'].tolist()))
        licks_R = list(chain.from_iterable(beh_df['licksR'].tolist()))

        beh_df.drop(['allLicks', 'licksL', 'licksR'], axis=1, inplace=True)


    else:
        print("'beh' is not a struct or has unexpected format.")
    return beh_df, licks_L, licks_R

def load_session_df(session):
    """
    Load the session data from the .mat file.

    Args:
        session (str): The session name, e.g., 'mBB041d20161006'

    Returns:
        pd.DataFrame: The session data.
    """
    path_data = parse_session_string(session)
    session_df, licks_L, licks_R = load_df_from_mat(os.path.join(path_data["sortedFolder"], f"{session}_sessionData_behav.mat"))
    return session_df, licks_L, licks_R
