# %%
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.append('..') 
import platform
import os
from pathlib import Path
import shutil
from pathlib import Path
import shutil
from utils.behavior.session_utils import load_session_df, parse_session_string, load_df_from_mat, load_neurons_from_mat
from utils.behavior.lick_analysis import clean_up_licks, parse_lick_trains
from utils.ephys.neuron_utils import load_neuron_df, load_neurons_from_mat
from scipy.io import loadmat
from itertools import chain
from matplotlib import pyplot as plt
from IPython.display import display
from scipy.signal import find_peaks
from harp.clock import align_timestamps_to_anchor_points
from utils.basics.data_org import *
from utils.photometry.preprocessing import * 
from utils.photometry.plot_utils import align_signal_to_events, plot_FP_with_licks, color_gradient
import numpy as np
from scipy.signal import butter, filtfilt, medfilt
from scipy.optimize import curve_fit
import json
from sklearn.linear_model import LinearRegression
from matplotlib.gridspec import GridSpec
import pickle
# %matplotlib inline
# %matplotlib widget

# %%
root = curr_computer()
anis = ['ZS059', 'ZS060', 'ZS061', 'ZS062']
for ani in anis: 
    opto_units_spreadsheet = pd.read_excel(os.path.join(root, ani+'.xlsx'), sheet_name='neurons')
    session_list = opto_units_spreadsheet['session'].unique().tolist()
    for session in session_list:
        path_data = parse_session_string(session)
        opto_neuron_df = opto_units_spreadsheet[(opto_units_spreadsheet['session'] == session)].copy()
        opto_neuron_df.reset_index(drop=True, inplace=True)
        _, _, _, neuron_df = load_neuron_df(session)
        opto_neuron_df['spike_times'] = None
        for index, id in enumerate(opto_neuron_df['ID']):
            # add spike times to the opto_neuron_df
            # spike_times = neuron_df[neuron_df['unit'] == id]['spike_times'].values[0]
            opto_neuron_df.at[index, 'spike_times'] = neuron_df[neuron_df['unit'] == id]['spike_times'].values
            # load quality metrics
            if opto_neuron_df.loc[index, 'Lratio']<=0.05:
                file_name = f"{session}_{id}_met.mat"
                unit_qm_mat = os.path.join(path_data['nlynxFolderSession'], file_name)
                unit_qm = loadmat(unit_qm_mat)
                unit_qm['met'].dtype.names

                for key in unit_qm['met'].dtype.names:
                    if key == 'width':
                        new_key = 'width_opto'
                    else:
                        new_key = key
                    if new_key not in opto_neuron_df.columns:
                        opto_neuron_df[new_key] = None
                    opto_neuron_df.at[index, new_key] = np.squeeze(unit_qm['met'][key][0][0])
                if 'metSess' in unit_qm.keys():
                    for key in unit_qm['metSess'].dtype.names:
                        if key == 'width':
                            new_key = 'width_session'
                        else:
                            new_key = key
                        if new_key not in opto_neuron_df.columns:
                            opto_neuron_df[new_key] = None
                        opto_neuron_df.at[index, new_key] = np.squeeze(unit_qm['metSess'][key][0][0])    
        save_file = os.path.join(path_data['sortedFolder'], f"{session}_opto_units.pkl")
        opto_neuron_df.drop(columns=['ID.1'], inplace=True)
        opto_neuron_df.rename(columns={'ID': 'unit'}, inplace=True)
        with open(save_file, 'wb') as f:
            pickle.dump(opto_neuron_df, f)

# opto_units_spreadsheet.to_excel(os.path.join(root, ani+'_opto_units.xlsx'), index=False)          


