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
from scipy.signal import butter, filtfilt, medfilt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
import pickle

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
        print(f"Region {value} recorded at fiber {key}")
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

def double_exp(x, start_values):
    """
    Double exponential function: a * exp(-b * x) + c * exp(-d * x) + e
    """
    return start_values[0] * np.exp(-start_values[1] * x) + start_values[2] * np.exp(-start_values[3] * x) + start_values[4]
def single_exp(x, start_values):
    """
    Single exponential function: a * exp(-b * x) + c
    """
    return start_values[0] * np.exp(-start_values[1] * x) + start_values[2]
def double_exp_fit(x, y, start_values):
    """
    Perform a double exponential fit to the given data.

    Parameters:
        x (array): Independent variable data.
        y (array): Dependent variable data.
        start_values (list): Initial guesses for the parameters [a, b, c, d, e].

    Returns:
        fit_params (array): Optimized parameters of the fit [a, b, c, d, e].
        gof (dict): Goodness-of-fit metrics, including R-squared and residuals.
    """
    # Perform the curve fitting
    popt, pcov = curve_fit(
        lambda x, a, b, c, d, e: double_exp(x, [a, b, c, d, e]), x, y, p0=start_values, bounds=(0, np.inf)
    )

    # Calculate goodness-of-fit metrics
    residuals = y - double_exp(x, popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)

    gof = {
        "R-squared": r_squared,
        "SS_res": ss_res,
        "SS_tot": ss_tot
    }

    return popt, gof

def single_exp_fit(x, y, start_values):
    """
    Perform a single exponential fit to the given data.

    Parameters:
        x (array): Independent variable data.
        y (array): Dependent variable data.
        start_values (list): Initial guesses for the parameters [a, b, c].

    Returns:
        fit_params (array): Optimized parameters of the fit [a, b, c].
        gof (dict): Goodness-of-fit metrics, including R-squared and residuals.
    """
    # Perform the curve fitting
    popt, pcov = curve_fit(
        lambda x, a, b, c: single_exp(x, [a, b, c]), x, y, p0=start_values, bounds=(0, np.inf)
    )

    # Calculate goodness-of-fit metrics
    residuals = y - single_exp(x, popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)

    gof = {
        "R-squared": r_squared,
        "SS_res": ss_res,
        "SS_tot": ss_tot
    }

    return popt, gof

def triple_exp(x, params):
    """
    Triple exponential function: a * exp(-b * x) + c * exp(-d * x) + f * exp(-g * x) + e
    """
    return (
        params[0] * np.exp(-params[1] * x) +
        params[2] * np.exp(-params[3] * x) +
        params[4] * np.exp(-params[5] * x) +
        params[6]
    )

def triple_exp_fit(x, y, start_values):
    """
    Perform a triple exponential fit to the given data.

    Parameters:
        x (array): Independent variable data.
        y (array): Dependent variable data.
        start_values (list): Initial guesses for the parameters [a, b, c, d, f, g, e].

    Returns:
        fit_params (array): Optimized parameters of the fit [a, b, c, d, f, g, e].
        gof (dict): Goodness-of-fit metrics, including R-squared and residuals.
    """
    # Perform the curve fitting
    
    popt, pcov = curve_fit(
        lambda x, a, b, c, d, f, g, e: triple_exp(x, [a, b, c, d, f, g, e]), x, y, p0=start_values, maxfev=10000, bounds=(0, np.inf)
    )

    # Calculate goodness-of-fit metrics
    residuals = y - triple_exp(x, popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)

    gof = {
        "R-squared": r_squared,
        "Residuals": residuals,
        "SS_res": ss_res,
        "SS_tot": ss_tot
    }

    return popt, gof

def denoising(raw_trace, fs, fc, plot = False):
    """
    Denoise the raw trace signal using filtering and exponential decay removal.

    Parameters:
        raw_trace (array): The raw input signal.
        fs (float): Sampling frequency in Hz.
        fc (float): Cutoff frequency for low-pass filtering in Hz.

    Returns:
        denoised (array): The denoised signal.
    """

    # Low-pass filter
    b, a = butter(2, fc / (fs / 2), btype='low')
    low_passed = filtfilt(b, a, raw_trace)

     # Exponential decay (bleaching) removal
    start_values = np.zeros(7)

    # Last 1 minute
    start_values[4] = np.mean(low_passed[-int(60 * fs):])

    # Major decay a
    diff_start = np.mean(low_passed[:int(60 * fs)]) - np.mean(low_passed[-int(60 * fs):])
    diff_middle = np.mean(low_passed[int(60 * fs):int(120 * fs)]) - np.mean(low_passed[-int(60 * fs):])
    start_values[1] = np.log(diff_start / diff_middle) / 60 if diff_middle != 0 else 0

    # Minor decay c
    start_values[3] = start_values[1] / 3

    # Scaling factors
    start_values[0] = 0.5 * diff_start
    start_values[2] = 0.5 * diff_start

    # start_values = [1, 1, 1, 0.05, 0.1]
    start_values[4] = 0.02
    start_values[5] = 0.1
    start_values[6] = np.mean(low_passed[-60*fs:-1])

        
    # Time array
    time = np.arange(len(low_passed)) / fs

    fit_params, _ = triple_exp_fit(time, low_passed, start_values)
    decay = triple_exp(time, fit_params)

    # # single exponential fit
    # start_values = [1, 0.05, 0.1]

    # # Ensure start_values are valid
    # start_values = np.real(start_values)
    # start_values[start_values < 0] = 0

    # # Fit and remove exponential decay
    # fit_params, _ = single_exp_fit(time, low_passed, start_values)
    # decay = single_exp(time, fit_params)
 
    bleach_removed = (low_passed - decay)/decay


    # High-pass filter to account for slow decay
    fc2 = 0.01  # Cutoff frequency in Hz
    b2, a2 = butter(2, fc2 / (fs / 2), btype='high')
    high_passed = filtfilt(b2, a2, bleach_removed)
    denoised = bleach_removed
    if plot:
        fig, ax = plt.subplots(4, 1, figsize=(20, 8))
        ax[0].plot(time, raw_trace, label='Raw')
        params_for_display = ', '.join([f'{param:.2f}' for param in fit_params])
        ax[0].set_title(f'Raw Signal {params_for_display}')
        ax[0].plot(time, low_passed, label='Low-pass')
        # ax[0].set_title('Low-pass Filtered')
        ax[0].plot(time, decay, label='Decay')
        ax[0].legend()
        ax[1].plot(time, low_passed - decay, label='Bleach Removed')
        ax[1].legend()
        # ax2 = ax[1].twinx()
        ax[2].plot(time, bleach_removed, label='Bleach normalized')
        ax[2].legend()
        ax[3].plot(time, high_passed, label='High-pass Filtered')
        ax[3].legend()
        plt.tight_layout()
        return denoised, fig
    else:
        return denoised

def preprocess_signal(session_id, signal_region_raw, fs = 20, lowcut = 0.1, fc = 9, baseline_remove = 20, plot=False, plot_len = None):
    session_dir = parse_session_string(session_id)
    signal_region_prep = {}
    for channel in signal_region_raw.keys():
        curr_channel = {}
        if 'time' not in channel:
            for region in signal_region_raw[channel].keys():
                print(f'Preprocessing {channel}{region}')
                signal = signal_region_raw[channel][region]
                # Denoising
                signal = signal[baseline_remove*fs:]
                if plot:
                    denoised, fig = denoising(signal, fs, fc, plot = plot)
                    fig.savefig(os.path.join(session_dir['saveFigFolder'], f'{session_id}_{channel}_{region}_denoised.pdf'))
                else:
                    denoised = denoising(signal, fs, fc, plot = plot)
                denoised = np.concatenate((np.full(baseline_remove*fs, np.nan), denoised))
                curr_channel[region] = denoised
            signal_region_prep[channel] = curr_channel
    
    G_Iso = {}
    Iso_fit = {}
    for region in signal_region_raw['G'].keys():
        tmp_G = signal_region_prep['G'][region]
        tmp_Iso = signal_region_prep['Iso'][region]
        clean_inds = np.where(~np.isnan(tmp_G) & ~np.isnan(tmp_Iso))
        lm = LinearRegression()
        lm.fit(tmp_Iso[clean_inds].reshape(-1, 1), tmp_G[clean_inds])
        G_Iso[region] = zscore(tmp_G[clean_inds] - lm.predict(tmp_Iso[clean_inds].reshape(-1, 1)))
        G_Iso[region] = np.concatenate((np.full(baseline_remove*fs, np.nan), G_Iso[region]))
        Iso_fit[region] = lm.predict(tmp_Iso[clean_inds].reshape(-1, 1))
        Iso_fit[region] = np.concatenate((np.full(baseline_remove*fs, np.nan), Iso_fit[region]))
    signal_region_prep['G-Iso'] = G_Iso
    signal_region_prep['Iso_fit'] = Iso_fit
    signal_region_prep['time_in_beh'] = signal_region_raw['time_in_beh']
    signal_region_prep['time'] = signal_region_raw['time']  
    
    if plot:
        fig, ax = plt.subplots(3, len(signal_region_prep['G'].keys()), figsize=(10, 5))
        for region_ind, region in enumerate(signal_region_prep['G'].keys()):
            ax[0, region_ind].plot(signal_region_prep['time_in_beh'], signal_region_prep['G'][region], label='G', linewidth=0.5)
            ax[0, region_ind].plot(signal_region_prep['time_in_beh'], signal_region_prep['Iso_fit'][region], label='Iso_fit', linewidth=0.5)
            ax[0, region_ind].set_title(region)
            ax[0, region_ind].legend()
            ax[1, region_ind].plot(signal_region_raw['time_in_beh'], signal_region_prep['Iso'][region], label='Iso', linewidth=0.5)
            ax[1, region_ind].legend()
            ax[2, region_ind].plot(signal_region_raw['time_in_beh'], signal_region_prep['G-Iso'][region], label='G-Iso', linewidth=0.5)
            ax[2, region_ind].legend()
            ax[2, region_ind].set_xlabel('Time (ms)')
        if plot_len is not None:
            plt.setp(ax, xlim=[signal_region_prep['time_in_beh'][0]+baseline_remove*1000, signal_region_prep['time_in_beh'][0]+baseline_remove*1000+plot_len*1000])
        else:
            plot_len = 'whole_session'
        # Setting the values for all axes.
        plt.suptitle('Preprocessing')
        plt.tight_layout()
        fig.savefig(os.path.join(session_dir['saveFigFolder'], f'{session_id}_preprocessed_{plot_len}.pdf'))
        
        return signal_region_prep, fig

    return signal_region_prep

def load_session_FP_raw(session, label, channels = ['G', 'Iso'], plot = False):
    _, licks_L, licks_R = load_session_df(session)
    session_dir = parse_session_string(session)
    signal_region = load_session_FP(session, label, plot=False)
    # Directory containing the files
    fpDir = session_dir['photometryPath']
    photometry_json = os.path.join(session_dir['photometryPath'], f'{session}.json')
    with open(photometry_json, "r") as file:
        location_info = json.load(file)
    # Get all files in the directory
    allFiles = os.listdir(fpDir)

    # Filter only files (ignore directories)
    allFiles = [f for f in allFiles if not os.path.isdir(os.path.join(fpDir, f))]
    time_stamps = None
    signal_region_raw = {}
    # Load signal files
    for channel in channels: 
        curr_sig = {}
        channelInd = [f for f in allFiles if f'FIP_Data{channel}' in f]
        if len(channelInd) > 0:
            channelSigData = pd.read_csv(os.path.join(fpDir, channelInd[0]), header=None).to_numpy()
            if channel == 'G':
                time_stamps = channelSigData[:, 0]
            # curr_sig[channel] = channelSigData[:, 1:1 + len(location_info)]
            for key, value in location_info.items():
                print(f"Channel {channel}:Region {value} recorded at fiber {key}")
                curr_sig[value] = np.array(channelSigData[:, int(key)+1])
        signal_region_raw[channel] = curr_sig

    for color in channels:
        curr_sig = signal_region_raw[color]
        for key, value in location_info.items():
            curr_sig[value] = curr_sig[value][(time_stamps <= signal_region['time'][-1]) & (time_stamps >= signal_region['time'][0])]


    signal_region_raw['time'] = time_stamps
    signal_region_raw['time_in_beh'] = signal_region['time_in_beh']

    if plot:
        fig, ax = plt.subplots(len(location_info.keys()), len(channels), figsize=(10, 5))
        for i, key in enumerate(location_info.keys()):
            for j, color in enumerate(channels):
                ax[i, j].plot(signal_region_raw['time_in_beh'], signal_region_raw[color][location_info[key]], label=color)
                ax[i, j].set_title(f'{color}{key}')
                if j == 0 and i == 0:
                    ax2 = ax[i, j].twinx()
                    ax2.hist(licks_L, bins=100, alpha=0.5, label='Licks L')
                    ax2.hist(licks_R, bins=100, alpha=0.5, label='Licks R')            
                if j == 0:
                    ax[i, j].set_ylabel(location_info[key])
                if i == len(location_info.keys()) - 1:
                    ax[i, j].set_xlabel('Time (ms)')
                ax[i, j].legend()
        plt.suptitle('Alignment Check')
        plt.tight_layout()
        fig.savefig(os.path.join(session_dir['saveFigFolder'], f'{session}_alignment_check.pdf'))
        return signal_region_raw, fig
    else:
        return signal_region_raw

def get_FP_data(session, label, save=True):
    session_dir = parse_session_string(session)
    if os.path.exists(os.path.join(session_dir['sortedFolder'], f'{session}_FP_{label}.pkl')):
        with open(os.path.join(session_dir['sortedFolder'], f'{session}_FP_{label}.pkl'), 'rb') as f:
            signal_region_prep = pickle.load(f)
        print(f'Loaded {session}_FP_{label}.pkl')
    else:
        signal_region_raw, _ = load_session_FP_raw(session, label, channels = ['G', 'Iso'], plot = True)
        signal_region_prep, _ = preprocess_signal(session, signal_region_raw, plot=True)
        if save:
            save_FP(signal_region_prep, session, tag = label)
        print(f'Created {session}_FP_{label}.pkl')
    return signal_region_prep

def save_FP(preprocessed_signal, session, tag = 'regular'):
    session_dir = parse_session_string(session)
    with open(os.path.join(session_dir['sortedFolder'], f'{session}_FP_{tag}.pkl'), 'wb') as f:
        pickle.dump(preprocessed_signal, f)