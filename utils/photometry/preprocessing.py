import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.append('..') 
import platform
import os
from pathlib import Path
import shutil
from utils.basics.data_org import curr_computer, move_subfolders
from pathlib import Path
import shutil
from utils.behavior.session_utils import load_session_df, parse_session_string
from utils.behavior.lick_analysis import clean_up_licks, parse_lick_trains
from scipy.io import loadmat
from itertools import chain
from matplotlib import pyplot as plt
from IPython.display import display
from scipy.signal import find_peaks
from harp.clock import align_timestamps_to_anchor_points
import json

def load_session_FP(session, label, plot=False):
    session_df, licks_L, _ = load_session_df(session)
    session_path = parse_session_string(session)
    file_name = f'{session}_photometry{label}.mat'
    photometry_file = os.path.join(session_path['sortedFolder'], file_name)
    photometry_json = os.path.join(session_path['photometryPath'], f'{session}.json')
    signal_mat = loadmat(photometry_file)
    with open(photometry_json, "r") as file:
        location_info = json.load(file)
    dFF = signal_mat['dFF']    
    # load photometry data and align to behavior
    signal_region = {}
    for key, value in location_info.items():
        print(f"Region {value} recorded at channel {key}")
        signal_region[value] = np.array(dFF[int(key)][0])
    signal_region['time'] = np.squeeze(np.array(signal_mat['timeFIP']))
    signal_region['time_in_beh'] = align_timestamps_to_anchor_points(
        signal_region['time'],
        np.array(signal_mat['trialStarts'][0]),
        session_df['CSon'].values
    )
    if plot:
        fig, ax = plt.subplots()
        ax.plot(signal_region['time_in_beh'], signal_region[location_info['0']], label='channel 0')
        ax2 = ax.twinx()
        ax2.hist(licks_L, bins=100, alpha=0.5, label='Licks L')
        ax.set_title('Alignment Check')
        plt.show()
        return signal_region, fig
    else:
        return signal_region
