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
    



