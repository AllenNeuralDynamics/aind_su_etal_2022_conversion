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
from utils.basics.data_org import curr_computer

def generate_session_data_operant_matching_decoupled_rwd_delay(session_name, save_flag=1):
    # Set paths
    root, sep = curr_computer()  # Function to determine if the system is PC or Mac
    animal_name, date = session_name.split('d')[0][1:], session_name.split('d')[1][:9]
    session_folder = f"m{animal_name}{date}"
    filepath = os.path.join(root, animal_name, session_folder, 'behavior', f"{session_name}.asc")

    # Determine save path
    if session_name[-1].isalpha():
        savepath = os.path.join(root, animal_name, session_folder, 'sorted', f"session {session_name[-1]}")
    else:
        savepath = os.path.join(root, animal_name, session_folder, 'sorted', 'session')

    # Import session data
    session_text = import_data_operant_matching(filepath)

    beh_session_data = {
        'trialType': [], 'trialEnd': [], 'CSon': [], 'licksL': [], 'licksR': [],
        'rewardL': [], 'rewardR': [], 'respondTime': [], 'rewardTime': [],
        'rewardProbL': [], 'rewardProbR': [], 'allLicks': [], 'laser': []
    }

    block_switch = [1]
    block_switch_l = [1]
    block_switch_r = [1]

    for i, line in enumerate(session_text):
        if 'L Trial ' in line:  # trial start
            curr_trial = int(re.search(r'\((\d+)\)', line).group(1))  # extract current trial number
            beh_session_data[curr_trial]['laser'] = 0

            # Find CS onset
            t_begin = i
            t_cs_flag = False
            while not t_cs_flag:
                if 'CS ' in session_text[i]:
                    t_cs = i
                    t_cs_flag = True
                i += 1

            # Find trial end
            t_end_flag = False
            while not t_end_flag:
                if 'CS ' in session_text[i]:
                    t_end = i - 2
                    t_end_flag = True
                i += 1
                if i == len(session_text):
                    t_end = len(session_text)
                    t_end_flag = True

            # Process trial data
            water_deliver_flag = False
            all_l_licks = []
            all_r_licks = []

            if 'CS PLUS' in session_text[t_cs]:
                beh_session_data[curr_trial]['trialType'] = 'CSplus'
                beh_session_data[curr_trial]['CSon'] = float(re.split(': ', session_text[t_cs])[1])
            elif 'CS MINUS' in session_text[t_cs]:
                beh_session_data[curr_trial]['trialType'] = 'CSminus'
                beh_session_data[curr_trial]['CSon'] = float(re.split(': ', session_text[t_cs])[1])

            for trial_idx in range(t_begin, t_cs):
                if 'Contingency' in session_text[trial_idx]:
                    temp = re.split(r'). ', session_text[trial_idx])
                    reward_prob_l, reward_prob_r = map(float, re.split(r'/', temp[1]))
                    beh_session_data[curr_trial]['rewardProbL'] = reward_prob_l
                    beh_session_data[curr_trial]['rewardProbR'] = reward_prob_r

            for trial_idx in range(t_cs, t_end):
                if 'L: ' in session_text[trial_idx]:
                    all_l_licks.append(float(re.split(': ', session_text[trial_idx])[1]))
                    beh_session_data[0]['allLicks'].append(float(re.split(': ', session_text[trial_idx])[1]))
                elif 'R: ' in session_text[trial_idx]:
                    all_r_licks.append(float(re.split(': ', session_text[trial_idx])[1]))
                    beh_session_data[0]['allLicks'].append(float(re.split(': ', session_text[trial_idx])[1]))

                # Check for water delivery
                if not water_deliver_flag:
                    if 'WATER L DELIVERED' in session_text[trial_idx]:
                        beh_session_data[curr_trial]['rewardL'] = 1
                        beh_session_data[curr_trial]['rewardR'] = np.nan
                        beh_session_data[curr_trial]['rewardTime'] = float(re.split(': ', session_text[trial_idx])[1])
                        water_deliver_flag = True
                    elif 'WATER L NOT DELIVERED' in session_text[trial_idx]:
                        beh_session_data[curr_trial]['rewardL'] = 0
                        beh_session_data[curr_trial]['rewardR'] = np.nan
                        beh_session_data[curr_trial]['rewardTime'] = float(re.split(': ', session_text[trial_idx])[1])
                        water_deliver_flag = True
                    elif 'WATER R DELIVERED' in session_text[trial_idx]:
                        beh_session_data[curr_trial]['rewardR'] = 1
                        beh_session_data[curr_trial]['rewardL'] = np.nan
                        beh_session_data[curr_trial]['rewardTime'] = float(re.split(': ', session_text[trial_idx])[1])
                        water_deliver_flag = True
                    elif 'WATER R NOT DELIVERED' in session_text[trial_idx]:
                        beh_session_data[curr_trial]['rewardR'] = 0
                        beh_session_data[curr_trial]['rewardL'] = np.nan
                        beh_session_data[curr_trial]['rewardTime'] = float(re.split(': ', session_text[trial_idx])[1])
                        water_deliver_flag = True

                if 'LASER' in session_text[trial_idx]:
                    beh_session_data[curr_trial]['laser'] = 1

                # End trial
                if trial_idx == t_end:
                    beh_session_data[curr_trial]['licksL'] = all_l_licks
                    beh_session_data[curr_trial]['licksR'] = all_r_licks
                    if all_l_licks or all_r_licks:
                        beh_session_data[curr_trial]['respondTime'] = min(
                            [l for l in all_l_licks if l >= beh_session_data[curr_trial]['CSon']] +
                            [r for r in all_r_licks if r >= beh_session_data[curr_trial]['CSon']]
                        )
                    else:
                        beh_session_data[curr_trial]['respondTime'] = np.nan

                    if not water_deliver_flag:
                        beh_session_data[curr_trial]['rewardL'] = np.nan
                        beh_session_data[curr_trial]['rewardR'] = np.nan
                        beh_session_data[curr_trial]['rewardTime'] = np.nan

                    if t_end != len(session_text):
                        beh_session_data[curr_trial]['trialEnd'] = float(re.split(': ', session_text[t_end + 2])[1])
                    else:
                        beh_session_data[curr_trial]['trialEnd'] = np.nan

            # Detect block switch
            if 'L Block Switch at Trial ' in session_text[trial_idx]:
                if curr_trial != 1:
                    block_switch.append(curr_trial + 1)
                    block_switch_l.append(curr_trial + 1)
            if 'R Block Switch at Trial ' in session_text[trial_idx]:
                if curr_trial != 1:
                    block_switch.append(curr_trial + 1)
                    block_switch_r.append(curr_trial + 1)

    # Trim block switch lists
    block_switch = [b for b in block_switch if b < len(beh_session_data)]
    block_switch_l = [b for b in block_switch_l if b < len(beh_session_data)]
    block_switch_r = [b for b in block_switch_r if b < len(beh_session_data)]

    # Save session data if required
    if save_flag:
        os.makedirs(savepath, exist_ok=True)
        save_file = os.path.join(savepath, f"{session_name}_sessionData_behav.mat")
        pd.DataFrame(beh_session_data).to_pickle(save_file)

    return beh_session_data, block_switch, block_switch_l, block_switch_r

def import_data_operant_matching(filepath):
    with open(filepath, 'r') as file:
        return file.readlines()

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


def load_session_data(path):
    # Placeholder for MATLAB .mat file loading
    return []

