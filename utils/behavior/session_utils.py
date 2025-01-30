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
    # lick_rate = calculate_lick_rate(beh_session_data, rwd_delay)
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
    if 'behSessionData' in mat_data:
        beh = mat_data['behSessionData']
    elif 'sessionData' in mat_data:
        beh = mat_data['sessionData']
    else:
        print("No 'behSessionData' or 'sessionData' found in the .mat file.")

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
                curr_list = [np.float64(x[0][0]) if x.shape[0] > 0 and not np.isnan(x[0][0]) else np.nan for x in curr_list]
            elif column in ['licksL', 'licksR', 'trialType']:
                curr_list = beh_df[column].tolist()
                curr_list = [x[0] if len(x)>0 else x for x in curr_list] 
            beh_df[column] = curr_list
        # all licks
        licks_L = list(chain.from_iterable(beh_df['licksL'].tolist()))
        licks_R = list(chain.from_iterable(beh_df['licksR'].tolist()))

        list_to_drop = ['licksL', 'licksR', 'allLicks', 'allSpikes']
        unit_list = [x for x in beh_df.columns if 'TT' in x]
        list_to_drop.extend(unit_list)

        beh_df.drop(list_to_drop, axis=1, inplace=True, errors='ignore')

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

