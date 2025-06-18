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
from scipy.signal import butter, filtfilt, medfilt, sosfiltfilt
from harp.clock import align_timestamps_to_anchor_points
import json
from scipy.signal import butter, filtfilt, medfilt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
import pickle
from aind_fip_dff.utils.preprocess import batch_processing, tc_triexpfit
from matplotlib.gridspec import GridSpec

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
        session_df['CSon'].values.astype(float)
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

def double_exp(x, start_values, fs = 20):
    """
    Double exponential function: a * exp(-b * x) + c * exp(-d * x) + e
    """
    if len(start_values) == 4:
        start_values = np.append(start_values, 0)
    if isinstance(x, (int)):
        x = np.arange(x) / fs
    return start_values[0] * np.exp(-start_values[1] * x) + start_values[2] * np.exp(-start_values[3] * x) + start_values[4]

def bright(T, start_values: np.ndarray, fs = 20) -> np.ndarray:
    """Baseline with  Biphasic exponential decay (bleaching)  x  increasing saturating exponential (brightening)"""
    if isinstance(T, (int)):
        T = -np.arange(T)
    return (
        start_values[0]
        * (
            1
            + start_values[1] * np.exp(T / (start_values[4] * fs))
            + start_values[2] * np.exp(T / (start_values[5] * fs))
        )
        * (1 - start_values[3] * np.exp(T / (start_values[6] * fs)))
    )

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

def single_exp_fit(x, y, start_values, xtol = 1e-8):
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
        lambda x, a, b, c: single_exp(x, [a, b, c]), x, y, p0=start_values, bounds=(0, np.inf), xtol = xtol
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

    return popt, pcov, gof

def triple_exp(x, params, fs = 20):
    """
    Triple exponential function: a * exp(-b * x) + c * exp(-d * x) + f * exp(-g * x) + e
    """
    if isinstance(x, (int)):
        x = np.arange(x) / fs   
    return (
        params[0] * np.exp(-params[1] * x) +
        params[2] * np.exp(-params[3] * x) +
        params[4] * np.exp(-params[5] * x) +
        params[6]
    )

def triple_exp_fit(x, y, start_values, xtol = 1e-8):
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
    # print(xtol)
    popt, pcov = curve_fit(
        lambda x, a, b, c, d, e, f, g: triple_exp(x, [a, b, c, d, e, f, g]), x, y, p0=start_values, maxfev=10000, bounds=(0, np.inf), xtol=xtol
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

def denoising(raw_trace, fs, fc, plot = False, xtol = 1e-8):
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
    start_values[6] = np.mean(low_passed[-5*fs:-1])
    start_values[start_values < 0] = 0

        
    # Time array
    time = 0.01 * np.arange(len(low_passed)) / fs 

    fit_params, _ = triple_exp_fit(time, low_passed, start_values, xtol = xtol)
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

def preprocess_signal(session_id, label, fs = 20, lowcut = 0.1, fc = 9, baseline_remove = 20, plot=False, plot_len = None, xtol = 1e-8, deep = False):
    session_dir = parse_session_string(session_id)
    signal_region_raw, _ = load_session_FP_raw(session_id, label, plot=True)
    signal_region_prep = {}
    if deep:
        for channel in signal_region_raw.keys():
            curr_channel = {}
            if 'time' not in channel:
                for region in signal_region_raw[channel].keys():
                    print(f'Preprocessing {channel}{region}')
                    signal = signal_region_raw[channel][region]
                    # Denoising
                    signal = signal[baseline_remove*fs:]
                    # check if exponential decay exists
                    params_fit, pcov, gof = single_exp_fit(0.01*np.arange(len(signal))/fs, signal, [1, 0.05, 0.1], xtol = 1e-6)
                    if signal[0]>1000:
                        print(f'Potential non-neuronal signal in {channel}{region}')
                    if params_fit[1]<1.00e-03:
                        denoised = signal
                        print(f'No exponential decay detected for {channel}{region}')
                    else:
                        if plot:
                            denoised, fig = denoising(signal, fs, fc, plot = plot, xtol=xtol)
                            fig.savefig(os.path.join(session_dir['saveFigFolder'], f'{session_id}_{channel}_{region}_denoised.pdf'))
                        else:
                            denoised = denoising(signal, fs, fc, plot = plot, xtol=xtol)
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
    else:
        signal_region_prep = signal_region_raw

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
            curr_sig[value] = curr_sig[value][(np.round(time_stamps, 2) <= np.round(signal_region['time'][-1], 2)) & (np.round(time_stamps,2) >= np.round(signal_region['time'][0],2))]


    signal_region_raw['time'] = signal_region['time']
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

def save_FP(preprocessed_signal, session, tag = 'regular'):
    session_dir = parse_session_string(session)
    with open(os.path.join(session_dir['sortedFolder'], f'{session}_FP_{tag}.pkl'), 'wb') as f:
        pickle.dump(preprocessed_signal, f)

def plot_FP_results(session, signal_region_prep, params, signal_region_raw = None, methods = ['exp', 'bright', 'tri_exp']):
    session_dir = parse_session_string(session)
    figs = []
    # all_fields = [field.split('_')[1] for field in list(signal_region_prep.keys()) if 'time' not in field and 'mc' not in field]
    if signal_region_raw is not None:
        signal_region_prep = signal_region_prep | signal_region_raw
    for method in methods:
        fig = plt.figure(figsize=(15, 8))
        gs = GridSpec(4, len(signal_region_prep['G'].keys()), figure=fig)
        for region_index, region in enumerate(signal_region_prep['G'].keys()):
            if method == 'bright':
                if len(params['G_bright'][region])==0:
                    baseline_G = np.zeros(len(signal_region_prep['time_in_beh']))
                    print(f'Fitting failed for {region} G with method {method}')
                else:
                    baseline_G = bright(len(signal_region_prep['time_in_beh']), params['G_bright'][region])
                if len(params['Iso_bright'][region])==0:
                    baseline_Iso = np.zeros(len(signal_region_prep['time_in_beh']))
                    print(f'Fitting failed for {region} Iso with method {method}')
                else:
                    baseline_Iso = bright(len(signal_region_prep['time_in_beh']), params['Iso_bright'][region])
            elif method == 'exp':
                if len(params['G_exp'][region])==0:
                    baseline_G = np.zeros(len(signal_region_prep['time_in_beh']))
                    print(f'Fitting failed for {region} G with method {method}')
                else:
                    baseline_G = double_exp(len(signal_region_prep['time_in_beh']), params['G_exp'][region])
                if len(params['Iso_exp'][region])==0:
                    baseline_Iso = np.zeros(len(signal_region_prep['time_in_beh']))
                    print(f'Fitting failed for {region} Iso with method {method}')
                else:
                    baseline_Iso = double_exp(len(signal_region_prep['time_in_beh']), params['Iso_exp'][region])
            elif method == 'tri-exp':
                if len(params['G_tri-exp'][region])==0:
                    baseline_G = np.zeros(len(signal_region_prep['time_in_beh']))
                    print(f'Fitting failed for {region} G with method {method}')
                else:
                    baseline_G = triple_exp(len(signal_region_prep['time_in_beh']), params['G_tri-exp'][region])
                if len(params['Iso_tri-exp'][region])==0:
                    baseline_Iso = np.zeros(len(signal_region_prep['time_in_beh']))
                    print(f'Fitting failed for {region} Iso with method {method}')
                else:
                    baseline_Iso = triple_exp(len(signal_region_prep['time_in_beh']), params['Iso_tri-exp'][region])
            ax = fig.add_subplot(gs[0, region_index])
            ax.plot(signal_region_prep['time_in_beh'], signal_region_prep['G'][region], linewidth = 0.5)
            ax.plot(signal_region_prep['time_in_beh'], baseline_G)
            ax.set_title(f'{region} G')
            ax = fig.add_subplot(gs[1, region_index])
            ax.plot(signal_region_prep['time_in_beh'], signal_region_prep['Iso'][region], linewidth = 0.5)
            ax.plot(signal_region_prep['time_in_beh'], baseline_Iso)
            ax.set_title(f'Iso')
            ax = fig.add_subplot(gs[2, region_index])
            ax.plot(signal_region_prep['time_in_beh'], signal_region_prep[f'G_{method}'][region], label = 'G', linewidth = 0.5)
            ax.legend()
            ax_flip = ax.twinx()
            ax_flip.plot(signal_region_prep['time_in_beh'], signal_region_prep[f'Iso_{method}'][region], label = 'Iso', color = 'r', linewidth = 0.5)
            ax_flip.legend()
            ax.set_title(f'dFF')
            ax = fig.add_subplot(gs[3, region_index])
            ax.plot(signal_region_prep['time_in_beh'], signal_region_prep[f'G_{method}_mc'][region], label = 'G', linewidth = 0.5)
            ax.set_title('mc')
        plt.suptitle(f'{session}_{region}_{method}')
        plt.tight_layout()
        fig.savefig(os.path.join(session_dir['saveFigFolder'], f'{session}_{method}_FP.pdf'))
        figs.append(fig) 
    return figs

def preprocess_signal_CO(session, label, fs = 20, methods = ['exp', 'bright', 'tri-exp'], baseline_remove = 10, plot=False, xtol = 1e-8):
    signal_region_raw = load_session_FP_raw(session, label, plot=False);
    signal_region_raw_CO = local_to_nwb_FP(session, signal_region_raw)
    df_fip_pp_nwb, df_PP_params, df_fip_mc = batch_processing(
        signal_region_raw_CO, methods, n_frame_to_cut=baseline_remove*fs, xtol=xtol
    )
    signal_region_prep = nwb_to_local_FP(df_fip_pp_nwb)
    signal_region_mc = nwb_to_local_FP(df_fip_mc)
    signal_region_mc= {f"{key}_mc":value for key, value in signal_region_mc.items() if 'time' not in key}
    signal_region_prep.update(signal_region_mc)
    params = nwb_to_local_FP_params(df_PP_params)
    figs = None
    if plot:
        figs = plot_FP_results(session, signal_region_prep, params, signal_region_raw = signal_region_raw, methods = methods)
    return signal_region_prep, params, figs

def local_to_nwb_FP(session, signal_region_raw):
    data = []
    for color, regions in signal_region_raw.items():
        if color != 'time' and color != 'time_in_beh':
            for region, signal in regions.items():
                for time, value in zip(signal_region_raw['time_in_beh'], signal):
                    data.append([region, color, time, value])

    # Create a dataframe
    signal_region_raw_CO = pd.DataFrame(data, columns=['fiber_number', 'channel', 'time_fip', 'signal'])
    signal_region_raw_CO.insert(0, "session", session)
    
    return signal_region_raw_CO

def nwb_to_local_FP(signal_region_raw_CO):
    signal_region_raw = {}
    for color in signal_region_raw_CO['channel'].unique():
        for method in signal_region_raw_CO['preprocess'].unique():
            signal_region_raw[f'{color}_{method}'] = {}
            for region in signal_region_raw_CO['fiber_number'].unique():
                signal_region_raw[f'{color}_{method}'][region] = signal_region_raw_CO[(signal_region_raw_CO['channel'] == color) & (signal_region_raw_CO['fiber_number'] == region) & (signal_region_raw_CO['preprocess'] == method)]['signal'].values
    signal_region_raw['time_in_beh'] = np.sort(signal_region_raw_CO['time_fip'].unique())
    return signal_region_raw

def nwb_to_local_FP_params(df_PP_params):
    is_number = [isinstance(item, (int, float)) for item in df_PP_params.columns] 
    numeric_columns = df_PP_params.columns[is_number]
    # Convert the selected columns into a matrix
    matrix = df_PP_params[numeric_columns].values
    params = {}
    for color in df_PP_params['channel'].unique():
        for method in df_PP_params['preprocess'].unique():
            params[f'{color}_{method}'] = {}
            for region in df_PP_params['fiber_number'].unique():
                params[f'{color}_{method}'][region] = matrix[(df_PP_params['channel'] == color) & (df_PP_params['fiber_number'] == region) & (df_PP_params['preprocess'] == method)]
                params[f'{color}_{method}'][region] = params[f'{color}_{method}'][region][0][~np.isnan(params[f'{color}_{method}'][region][0])]
    return params

def append_FP_data(session, label, signal_region_prep_new, df_PP_params):
    session_dir = parse_session_string(session)
    combined_pkl = os.path.join(session_dir['sortedFolder'], f'{session}_combined.pkl')
    combined_params_pkl = os.path.join(session_dir['sortedFolder'], f'{session}_combined_params.pkl')
    raw_pkl = os.path.join(session_dir['sortedFolder'], f'{session}_FP_{label}.pkl')

    if os.path.exists(combined_pkl):
        with open(combined_pkl, 'rb') as f:
            signal_region_prep = pickle.load(f)
        with open(combined_params_pkl, 'rb') as f:
            params = pickle.load(f)
        print(f'Loaded {session}_combined.pkl')
    else:
        with open(raw_pkl, 'rb') as f:
            signal_region_prep = pickle.load(f)
        params = {}
    signal_region_prep_updated = signal_region_prep | signal_region_prep_new
    # write to pickle
    with open(combined_pkl, 'wb') as f:
        pickle.dump(signal_region_prep_updated, f)
    print(f"Finished writing {session}_combined.pkl")
    params_updated = params | df_PP_params
    # write to pickle
    with open(os.path.join(session_dir['sortedFolder'], f'{session}_combined_params.pkl'), 'wb') as f:
        pickle.dump(params_updated, f)
    print(f"Finished writing {session}_combined_params.pkl")

def get_FP_data(session, label = None, save=True):
    session_dir = parse_session_string(session)
    if os.path.exists(os.path.join(session_dir['sortedFolder'], f'{session}_combined.pkl')) and os.path.exists(os.path.join(session_dir['sortedFolder'], f'{session}_combined_params.pkl')):
        with open(os.path.join(session_dir['sortedFolder'], f'{session}_combined.pkl'), 'rb') as f:
            signal_region_prep_updated = pickle.load(f)
        # open from .pkl
        with open(os.path.join(session_dir['sortedFolder'], f'{session}_combined_params.pkl'), 'rb') as f:
            params = pickle.load(f)
        print(f'Loaded {session}_combined.pkl and {session}_combined_params.pkl')
    elif os.path.exists(os.path.join(session_dir['sortedFolder'], f'{session}_FP_{label}.pkl')):
        with open(os.path.join(session_dir['sortedFolder'], f'{session}_FP_{label}.pkl'), 'rb') as f:
            signal_region_prep = pickle.load(f)
        print(f'Loaded {session}_FP_{label}.pkl')
        signal_region_prep_CO, params, _ = preprocess_signal_CO(session, label, plot=True, xtol=1e-6)
        signal_region_prep_updated = append_FP_data(session, label, signal_region_prep_CO, params)
        print(f'Appended CO version to {session}_combined.pkl')
    else:
        signal_region_raw, _ = load_session_FP_raw(session, label, channels = ['G', 'Iso'], plot = True)
        signal_region_prep = preprocess_signal(session, label, plot=False, xtol = 1e-6, deep = False)
        if save:
            save_FP(signal_region_prep, session, tag = label)
            print(f'Created {session}_FP_{label}.pkl')
        signal_region_prep_CO, params, _ = preprocess_signal_CO(session, label, plot=True, xtol=1e-6)
        # plot_FP_results(session, signal_region_prep_CO, params, methods = ['exp', 'bright', 'tri-exp'])
        signal_region_prep_updated = append_FP_data(session, label, signal_region_prep_CO, params)
        print(f'Appended CO version to {session}_combined.pkl')
    return signal_region_prep_updated, params


def peak_detect_FP(session, plot = False):
    session_dir = parse_session_string(session)
    signal_FP, _ = get_FP_data(session)
    sos = butter(2, 2, 'low', fs=20, output='sos')
    peaks = {}
    peak_amps = {}
    xlim_starts = np.linspace(signal_FP['time_in_beh'][0], signal_FP['time_in_beh'][-1], 4)
    xlim_starts = xlim_starts[:-1] + 20*1000
    xlim_ends = xlim_starts + 30*1000
    regions = signal_FP['G'].keys()
    if plot:
        fig, ax = plt.subplots(len(regions), len(xlim_starts)+1, figsize=(20, 5*len(regions)))

    for region_ind, region in enumerate(signal_FP['G'].keys()):
        curr_signal = signal_FP['G_tri-exp_mc'][region]
        curr_signal_filtered = sosfiltfilt(sos, curr_signal)
        # lower percentile of 10
        baseline = np.percentile(curr_signal_filtered, 10)
        height = baseline + 1*np.std(curr_signal_filtered)
        prominence = 1*np.std(curr_signal_filtered)
        distance = 20*0.5
        peaks[region], peak_amps[region] = find_peaks(curr_signal_filtered, height = height, distance = distance, prominence=prominence)
        peaks[region] = signal_FP['time_in_beh'][peaks[region]]
        peak_amps[region] = peak_amps[region]['peak_heights']
        if plot:
            bins = np.linspace(np.min(curr_signal_filtered), np.max(curr_signal_filtered), 50)
            ax[region_ind, 0].hist(peak_amps[region], bins = bins, alpha = 0.5, color = 'red', density = True, label = 'peak_amp')
            ax[region_ind, 0].hist(curr_signal_filtered, bins = bins, alpha = 0.5, color = 'k', density = True, label = 'all_signal')
            ax[region_ind, 0].axvline(height, color = 'blue', linestyle = '--', linewidth = 2, label = 'threshold')
            ax[region_ind, 0].axvline(baseline, color = 'black', linestyle = '--', linewidth = 2, label = 'baseline')
            ax[region_ind, 0].set_title(region)
            if region_ind == 0:
                ax[region_ind, 0].legend()
            for xlim_ind, (xlim_start, xlim_end) in enumerate(zip(xlim_starts, xlim_ends)):
                ax[region_ind, xlim_ind+1].plot(signal_FP['time_in_beh'], curr_signal)
                ax[region_ind, xlim_ind+1].plot(signal_FP['time_in_beh'], curr_signal_filtered, color = [0, 0, 0, 0.25])
                ax[region_ind, xlim_ind+1].scatter(peaks[region], peak_amps[region], color = 'red', zorder = 10)
                ax[region_ind, xlim_ind+1].axhline(height, color = 'black', linestyle = '--')
                ax[region_ind, xlim_ind+1].set_xlim(xlim_start, xlim_end)
                ax[region_ind, xlim_ind+1].set_xlabel('time (ms)')
            plt.suptitle(session)
    peaks_all = {'peak_time': peaks, 'peak_amplitude': peak_amps}    
    if plot:
        fig.savefig(os.path.join(session_dir["saveFigFolder"], f'peak_detect_FP.pdf'))                                   
        return peaks_all, fig
    else:
        return peaks_all
