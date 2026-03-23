import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.append('..') 
sys.path.append(r'C:\Users\zhixi\Documents\GitHub\aind-beh-ephys-analysis\code')
from beh_ephys_analysis.utils.ephys_functions import fitSpikeModelG
import platform
import os
from pathlib import Path
from utils.basics.data_org import curr_computer
import shutil
from utils.behavior.session_utils import load_session_df, parse_session_string, beh_analysis_no_plot
from utils.behavior.model_utils import get_stan_model_params_samps_only, infer_model_var
from utils.behavior.lick_analysis import clean_up_licks, parse_lick_trains
from utils.photometry.preprocessing import get_FP_data, peak_detect_FP
from utils.ephys.plot_utils import plot_raster_rate, plot_rate
from itertools import chain
from matplotlib import pyplot as plt
# from utils.photometry.preprocessing import get_FP_data
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import lfilter

def color_gradient(colors, num_points):
    # generate gradient given same number of points between two colors in a list of colors
    # colors: list of colors in RGB format
    # num_points: number of total points
    # return: list of colors in RGB format so that there's no color from colors in the gradient
    gradient = []
    for i in range(len(colors)-1):
        for j in range(1, num_points//(len(colors)-1)+1):
            gradient.append([colors[i][0] + j/(num_points//(len(colors)-1)+1)*(colors[i+1][0]-colors[i][0]), 
                             colors[i][1] + j/(num_points//(len(colors)-1)+1)*(colors[i+1][1]-colors[i][1]), 
                             colors[i][2] + j/(num_points//(len(colors)-1)+1)*(colors[i+1][2]-colors[i][2])])
    return gradient


def smooth_exp(y, fs, tau_s=0.2):
    from scipy.signal import lfilter

    y = np.asarray(y, float)
    out = np.full_like(y, np.nan)

    a = np.exp(-1.0/(fs*tau_s))
    b = [1-a]

    valid = ~np.isnan(y)

    idx = np.where(np.diff(np.r_[False, valid, False]))[0].reshape(-1,2)

    for start, end in idx:
        out[start:end] = lfilter(b, [1, -a], y[start:end])

    return out


def align_signal_to_events(signal, signal_time, event_times, kernel = False, avoid_other_events = True, tau = 0.2, pre_event_time=1000, post_event_time=2000, window_size=100, step_size=10, ax = None, legend = 'signal', color = 'b', plot_error = True):
    """
    Aligns the signal to event times and generates a matrix and PSTH using a moving average.

    Parameters:
    signal (np.ndarray): The signal to be aligned.
    signal_time (np.ndarray): The time points corresponding to the signal.
    event_times (np.ndarray): The times of the events to align to.
    pre_event_time (int): Time before the event to include in the alignment (in ms).
    post_event_time (int): Time after the event to include in the alignment (in ms).
    window_size (int): The size of the moving average window (in ms).
    step_size (int): The step size for the moving average (in ms).

    Returns:
    aligned_matrix (np.ndarray): The matrix of aligned signals.
    psth (np.ndarray): The Peri-Stimulus Time Histogram.
    time_bins (np.ndarray): The time bins for the PSTH.
    """
    
    
    # time_bins = np.arange(-pre_event_time, post_event_time - window_size + step_size, step_size)
    avoid_pre = 0
    avoid_post = 3000
    # filter signal with exponential kernel
    if kernel:
        signal = smooth_exp(signal, fs=20, tau_s=tau)
        signal[~np.isnan(signal)] = zscore(signal[~np.isnan(signal)])
        num_steps = (pre_event_time + post_event_time) // step_size + 1
        time_bins = -pre_event_time + np.array(range(num_steps)) * step_size
        aligned_matrix = np.zeros((len(event_times), num_steps))
    else:
        signal[~np.isnan(signal)] = zscore(signal[~np.isnan(signal)])
        num_steps = (pre_event_time + post_event_time - window_size) // step_size + 1
        time_bins = -pre_event_time + np.array(range(num_steps)) * step_size + window_size / 2
        aligned_matrix = np.zeros((len(event_times), num_steps))
    if not kernel:
        for i, event_time in enumerate(event_times):
            start_time = event_time - pre_event_time
            end_time = event_time + post_event_time
            mask = (signal_time >= start_time-0.5*window_size) & (signal_time < end_time+0.5*window_size)
            aligned_signal = signal[mask]
            aligned_time = signal_time[mask] - event_time

            for j in range(num_steps):
                window_start = j * step_size - pre_event_time
                window_end = window_start + window_size
                window_mask = (aligned_time >= window_start) & (aligned_time < window_end)
                aligned_matrix[i, j] = np.nanmean(aligned_signal[window_mask])
    else:
        # linear interpolation to get signal at exact time bins and get rid of timepoints too close to other events
        for i, event_time in enumerate(event_times):
            curr_bins = event_time + time_bins
            aligned_matrix[i, :] = np.interp(curr_bins, signal_time, signal, left=np.nan, right=np.nan)
            if avoid_other_events:
                next_event_time = event_times[i+1] if i < len(event_times)-1 else np.inf
                prev_event_time = event_times[i-1] if i > 0 else -np.inf
                valid_mask = (curr_bins > prev_event_time + avoid_post) & (curr_bins < next_event_time - avoid_pre)
                aligned_matrix[i, ~valid_mask] = np.nan

            


    mean_psth = np.nanmean(aligned_matrix, axis=0)
    std_psth = np.nanstd(aligned_matrix, axis=0)
    se_psth = std_psth / np.sqrt(aligned_matrix.shape[0])

    # Plotting the PSTH
    if ax is not None:
        ax.plot(time_bins, mean_psth, color=color, label=legend)
        ax.fill_between(time_bins, mean_psth - se_psth, mean_psth + se_psth, alpha=0.3, facecolor=color)
        if plot_error:
            ax.fill_between(time_bins, mean_psth - std_psth, mean_psth + std_psth, alpha=0.1, facecolor=color)
        ax.set_xlabel('Time (ms)')
    return aligned_matrix, mean_psth, time_bins, ax

def plot_FP_with_licks(session, label, region):
    session_df, licks_L, licks_R = load_session_df(session)
    session_dir = parse_session_string(session)
    signal_region_prep, params = get_FP_data(session, label)
    licks_L, licks_R, fig = clean_up_licks(licks_L, licks_R, plot=False)
    parsed_licks_L, _ = parse_lick_trains(licks_L)
    parsed_licks_R, _ = parse_lick_trains(licks_R)
    trial_starts = session_df['CSon']
    licks_in_trial_L = [train_start for train_start in list(parsed_licks_L['train_starts']) if any([trial_start<train_start and trial_start>train_start-2000  for trial_start in trial_starts])]
    licks_in_trial_R = [train_start for train_start in list(parsed_licks_R['train_starts']) if any([trial_start<train_start and trial_start>train_start-2000  for trial_start in trial_starts])]
    licks_out_trial_L = [train_start for train_start in list(parsed_licks_L['train_starts']) if not any([trial_start<train_start and trial_start>train_start-2000  for trial_start in trial_starts])]
    licks_out_trial_R = [train_start for train_start in list(parsed_licks_R['train_starts']) if not any([trial_start<train_start and trial_start>train_start-2000  for trial_start in trial_starts])]
    fig = plt.figure(figsize=(15, 40))
    colorL = 'b'
    colorR = 'r'
    all_channels = [key for key, value in signal_region_prep.items() if 'time' not in key]
    gs = GridSpec(len(all_channels), 5, figure=fig)
    for channel_id, channel in enumerate(all_channels):
        signal = signal_region_prep[channel][region]
        ax = fig.add_subplot(gs[channel_id, 0])
        align_signal_to_events(signal, signal_region_prep['time_in_beh'], parsed_licks_L['train_starts'], ax = ax, legend = 'L', color = colorL, avoid_other_events=False)
        align_signal_to_events(signal, signal_region_prep['time_in_beh'], parsed_licks_R['train_starts'], ax = ax, legend = 'R', color = colorR, avoid_other_events=False)
        ax.legend()
        ax.set_title(f'All licks')
        ax.set_ylabel(channel)
        # in vs out trial L
        ax = fig.add_subplot(gs[channel_id, 1])
        align_signal_to_events(signal, signal_region_prep['time_in_beh'], licks_in_trial_L, ax = ax, color = colorL, legend = 'in', avoid_other_events=False)
        align_signal_to_events(signal, signal_region_prep['time_in_beh'], licks_out_trial_L, ax = ax, color = colorR, legend = 'out', avoid_other_events=False)
        ax.legend()
        ax.set_title(f'In vs out trial L')
        # in vs out trial R
        ax = fig.add_subplot(gs[channel_id, 2])
        align_signal_to_events(signal, signal_region_prep['time_in_beh'], licks_in_trial_R, ax = ax, color = colorL, legend = 'in', avoid_other_events=False)     
        align_signal_to_events(signal, signal_region_prep['time_in_beh'], licks_out_trial_R, ax = ax, color = colorR, legend = 'out', avoid_other_events=False)
        ax.legend()
        ax.set_title(f'In vs out trial R')
        # in left, in vs out trial with gradient of lick lick peak
        ax = fig.add_subplot(gs[channel_id, 3])
        num_bins = 3
        peaks = parsed_licks_L['train_amps']
        colors_in = color_gradient([[1, 1, 1], [1, 0, 0]] , num_bins+1)
        colors_out = color_gradient([[1, 1, 1], [0, 0, 1]], num_bins+1)
        edges = np.quantile(peaks, np.linspace(0, 1, num_bins+1))
        for ind in range(num_bins):
            mask = (peaks>edges[ind]) & (peaks<=edges[ind+1])
            if np.sum(mask)>2:
                align_signal_to_events(signal, signal_region_prep['time_in_beh'], np.array(parsed_licks_L['train_starts'])[mask], ax = ax, color = colors_in[ind+1], legend = f'In trial bin {ind}', plot_error=False, avoid_other_events=False)
        ax.set_title(f'Left licks by lick peak')
        # in right, in vs out trial with gradient of lick lick peak\
        ax = fig.add_subplot(gs[channel_id, 4])
        peaks = parsed_licks_R['train_amps']
        colors_in = color_gradient([[1, 1, 1], [0, 0, 1]] , num_bins+1)
        colors_out = color_gradient([[1, 1, 1], [0, 0, 1]], num_bins+1)
        edges = np.quantile(peaks, np.linspace(0, 1, num_bins+1))
        edges[0] = edges[0]-0.01
        for ind in range(num_bins):
            mask = (peaks>edges[ind]) & (peaks<=edges[ind+1])
            if np.sum(mask)>2:
                align_signal_to_events(signal, signal_region_prep['time_in_beh'], np.array(parsed_licks_R['train_starts'])[mask], ax = ax, color = colors_in[ind+1], legend = f'In trial bin {ind}', plot_error=False, avoid_other_events=False)  
        ax.set_title(f'Right licks by lick peak')
    plt.suptitle(f'{session}_{region}')
    plt.tight_layout()

    fig.savefig(os.path.join(session_dir['saveFigFolder'], f'{session}_{region}_FP_licks.pdf'))

def plot_G_vs_Iso(session, zscore_flag = True):
    signal, _ = get_FP_data(session)
    session_df, licks_L, licks_R = load_session_df(session)
    parsed_licks_L, _ = parse_lick_trains(licks_L)
    parsed_licks_R, _ = parse_lick_trains(licks_R)
    session_dir = parse_session_string(session)
    regions = signal['G'].keys()
    start = signal['time_in_beh'][0]+100*1000
    fig = plt.figure(figsize=(20, 6))
    gs = GridSpec(len(regions), 4, height_ratios=[1]*len(regions), width_ratios=[5, 1, 1, 1])
    reward_times = session_df[(session_df['rewardL']==1) | (session_df['rewardR']==1)]['rewardTime']
    for region_ind, region in enumerate(regions):
        ax = plt.subplot(gs[region_ind, 0])
        if zscore_flag:
            ax.plot(signal['time_in_beh'], zscore(signal['G_tri-exp_mc'][region]), label='G_tri-exp_mc', linewidth=0.5, alpha=0.7)
            ax.plot(signal['time_in_beh'], zscore(signal['Iso_tri-exp_mc'][region]), label='Iso_tri-exp_mc', linewidth=0.5, alpha=0.7)
            peak = np.max(zscore(signal['G_tri-exp_mc'][region]))
        else:
            ax.plot(signal['time_in_beh'], signal['G_tri-exp_mc'][region], label='G_tri-exp_mc', linewidth=0.5, alpha=0.7)
            ax.plot(signal['time_in_beh'], signal['Iso_tri-exp_mc'][region], label='Iso_tri-exp_mc', linewidth=0.5, alpha=0.7)
            peak = np.max(signal['G_tri-exp_mc'][region])
        offset = 112136
        ax.set_xlim(start + offset, 120*1000+start + offset)
        if region_ind == 0:
            ax.legend()
        ax.scatter(session_df['CSon'], 1.3 * peak * np.ones_like(session_df['CSon']), c='k', s=5, label='go cue')
        # ax.scatter(parsed_licks_L['train_starts'], 1.2 * peak *np.ones_like(parsed_licks_L['train_starts']), c='r', s=5, label='left licks')
        # ax.scatter(parsed_licks_R['train_starts'], 1.2 * peak *np.ones_like(parsed_licks_R['train_starts']), c='b', s=5, label='right licks')
        # change to real licks
        ax.scatter(licks_L, 1.2 * peak * np.ones_like(licks_L), c='m', s=5, label='clean left licks', marker='|')
        ax.scatter(licks_R, 1.2 * peak * np.ones_like(licks_R), c='c', s=5, label='clean right licks', marker='|')
        ax.scatter(reward_times, 1.05 * peak * np.ones_like(reward_times), c='g', s=5, label='rewards', marker='D')
        ax.set_title(region)    
        ax.legend()

        from scipy.signal import welch

        fs = 20  # Sampling frequency (Hz)
        sig = signal['G'][region]
        sig = sig[~np.isnan(sig)]  # Remove NaN values for PSD calculation
        # Welch PSD (power per Hz)
        # nperseg: choose based on your data length; 256–2048 are common
        f, Pxx = welch(
            sig,
            fs=fs,
            window="hann",
            nperseg=min(256, len(sig)),
            noverlap=None,          # default = nperseg//2
            detrend="constant",
            scaling="density"       # PSD units: power/Hz
        )

        # (Optional) drop 0 Hz for log-log plotting
        mask = (f > 0) & (f < fs/2)  # Keep frequencies between 0 and Nyquist
        f = f[mask]
        Pxx = Pxx[mask]
  
        ax = plt.subplot(gs[region_ind, 1])
        ax.loglog(f, Pxx, label="Signal", linewidth=1, alpha=0.7, color='blue')
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power")
        # ax.grid(True, which="both", linestyle="--", alpha=0.6)
        # ax.set_xlim(0)

        # Plot Power Spectrum of Iso Signal
        sig = signal['Iso'][region]
        sig = sig[~np.isnan(sig)]  # Remove NaN values for PSD calculation
        f, Pxx = welch(
            sig,
            fs=fs,
            window="hann",
            nperseg=min(256, len(sig)),
            noverlap=None,          # default = nperseg//2
            detrend="constant",
            scaling="density"       # PSD units: power/Hz
        )
        mask = (f > 0) & (f < fs/2)  # Keep frequencies between 0 and Nyquist


        ax.loglog(f[mask], Pxx[mask], label="Iso Signal", linewidth=1, alpha=0.7, color='orange')
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power")
        ax.set_title("Power Spectrum G vs Iso")
        if region_ind == 0:
            ax.legend()
        # ax.set_xlim(0, 2)

        ax = plt.subplot(gs[region_ind, 2])
        align_signal_to_events(zscore(signal['G_tri-exp_mc'][region]), signal['time_in_beh'], session_df.loc[session_df['respondTime'].isna(), 'CSon'].values, ax=ax, color='r', legend='nogo')
        align_signal_to_events(zscore(signal['G_tri-exp_mc'][region]), signal['time_in_beh'], session_df.loc[~session_df['respondTime'].isna(), 'CSon'].values, ax=ax, color='b', legend='go')
        ax.set_title("G")
        ax.legend()

        ax = plt.subplot(gs[region_ind, 3])
        align_signal_to_events(zscore(signal['Iso_tri-exp_mc'][region]), signal['time_in_beh'], session_df.loc[session_df['respondTime'].isna(), 'CSon'].values, ax=ax, color='r', legend='nogo')
        align_signal_to_events(zscore(signal['Iso_tri-exp_mc'][region]), signal['time_in_beh'], session_df.loc[~session_df['respondTime'].isna(), 'CSon'].values, ax=ax, color='b', legend='go')
        ax.set_title("Iso")
        ax.legend()

        plt.tight_layout()

            

        # sos = butter(2, [0.1, 4], 'band', fs=20, output='sos')
        # G_low = sosfiltfilt(sos, signal['G_tri-exp_mc'][region])
        # Iso_low = sosfiltfilt(sos, signal['Iso_tri-exp_mc'][region])
        # plt.subplot(2, 1, 2, sharex=ax)
        # plt.plot(signal['time_in_beh'], G_low, label='G_tri-exp_mc')
        # plt.plot(signal['time_in_beh'], Iso_low, label='Iso_tri-exp_mc')
        # plt.xlim(start, 60*1000+start)
        # plt.legend()

        # lm = LinearRegression()
        # lm.fit(Iso_low.reshape(-1, 1), G_low.reshape(-1, 1))
        # lm.fit(Iso_low.reshape(-1, 1), G_low.reshape(-1, 1))
        # fitted_G = lm.predict(Iso_low.reshape(-1, 1))
        # lm = LinearRegression()
        # lm.fit(Iso_low.reshape(-1, 1), Iso_low.reshape(-1, 1))
        # fitted_Iso = lm.predict(Iso_low.reshape(-1, 1))
        fig.savefig(os.path.join(session_dir['saveFigFolder'], f'{session}_G_vs_Iso.pdf'))
        
    return fig


def plot_FP_beh_analysis(session, channel = 'G_tri-exp_mc'):
    signal, params = get_FP_data(session)
    regions = signal['G'].keys()
    session_df, licks_L, licks_R = load_session_df(session)
    session_dir = parse_session_string(session)
    reward_mask = (session_df['rewardR']>0) | (session_df['rewardL']>0)
    for region in regions:
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 5, height_ratios=[1, 1]) 
        curr_signal = zscore(signal[channel][region])
        # Aligned to cue
        # go cue vs nogo cue
        ax = fig.add_subplot(gs[0, 0])
        align_signal_to_events(curr_signal, signal['time_in_beh'], session_df.loc[(session_df['trialType']=='CSplus')&(~reward_mask), 'CSon'].values, legend = 'CSplus', color = 'b', plot_error = False, ax=ax)
        align_signal_to_events(curr_signal, signal['time_in_beh'], session_df.loc[(session_df['trialType']=='CSminus')&(~reward_mask), 'CSon'].values, legend = 'CSminus', color = 'r', plot_error = False, ax=ax)
        ax.set_xlabel('Time from go cue (ms)')
        ax.legend()
        # go vs no go
        ax = fig.add_subplot(gs[0, 1])
        align_signal_to_events(curr_signal, signal['time_in_beh'], session_df.loc[~session_df['respondTime'].isna(), 'CSon'].values, legend = 'Go', color = 'b', plot_error = False, ax=ax)
        align_signal_to_events(curr_signal, signal['time_in_beh'], session_df.loc[session_df['respondTime'].isna(), 'CSon'].values, legend = 'No Go', color = 'r', plot_error = False, ax=ax)
        ax.legend()
        ax.set_xlabel('Time from go cue (ms)')
        # go vs no go in go cue
        ax = fig.add_subplot(gs[0, 2])
        align_signal_to_events(curr_signal, signal['time_in_beh'], session_df.loc[(session_df['trialType']=='CSplus')&(~session_df['respondTime'].isna())&(~reward_mask), 'CSon'].values, legend = 'Go in CSplus', color = 'b', plot_error = False, ax=ax)
        align_signal_to_events(curr_signal, signal['time_in_beh'], session_df.loc[(session_df['trialType']=='CSplus')&(session_df['respondTime'].isna())&(~reward_mask), 'CSon'].values, legend = 'No Go in CSplus', color = 'r', plot_error = False, ax=ax)
        ax.legend()
        ax.set_xlabel('Time from go cue (ms)')
        # go vs no go in nogo cue
        ax = fig.add_subplot(gs[0, 3])
        
        align_signal_to_events(curr_signal, signal['time_in_beh'], session_df.loc[(session_df['trialType']=='CSminus')&(~session_df['respondTime'].isna())&(~reward_mask), 'CSon'].values, legend = 'Go in CSminus', color = 'b', plot_error = False, ax=ax)
        align_signal_to_events(curr_signal, signal['time_in_beh'], session_df.loc[(session_df['trialType']=='CSminus')&(session_df['respondTime'].isna())&(~reward_mask), 'CSon'].values, legend = 'No Go in CSminus', color = 'r', plot_error = False, ax=ax)
        ax.legend()
        ax.set_xlabel('Time from go cue (ms)')
        # Aligned to response
        # reward vs no reward
        ax = fig.add_subplot(gs[1, 0])
        align_signal_to_events(curr_signal, signal['time_in_beh'], session_df.loc[(session_df['rewardR']>0) | (session_df['rewardL']>0), 'respondTime'].values, legend = 'Reward', color = 'b', plot_error = False, ax=ax)
        align_signal_to_events(curr_signal, signal['time_in_beh'], session_df.loc[(session_df['rewardR']==0) | (session_df['rewardL']==0), 'respondTime'].values, legend = 'No Reward', color = 'r', plot_error = False, ax=ax)
        ax.legend()
        ax.set_xlabel('Time from response (ms)')
        # left vs right
        ax  = fig.add_subplot(gs[1, 1])
        align_signal_to_events(curr_signal, signal['time_in_beh'], session_df.loc[~session_df['rewardR'].isna(), 'respondTime'].values, legend = 'Right', color = 'b', plot_error = False, ax=ax)
        align_signal_to_events(curr_signal, signal['time_in_beh'], session_df.loc[~session_df['rewardL'].isna(), 'respondTime'].values, legend = 'Left', color = 'r', plot_error = False, ax=ax)
        ax.legend()
        ax.set_xlabel('Time from response (ms)')
        # reward vs no reward in left
        ax = fig.add_subplot(gs[1, 2])
        align_signal_to_events(curr_signal, signal['time_in_beh'], session_df.loc[session_df['rewardL']>0, 'respondTime'].values, legend = 'Reward in Left', color = 'b', plot_error = False, ax=ax)
        align_signal_to_events(curr_signal, signal['time_in_beh'], session_df.loc[session_df['rewardL']==0, 'respondTime'].values, legend = 'No Reward in Left', color = 'r', plot_error = False, ax=ax)
        ax.legend()
        ax.set_xlabel('Time from response (ms)')
        # reward vs no reward in right
        ax = fig.add_subplot(gs[1, 3])
        align_signal_to_events(curr_signal, signal['time_in_beh'], session_df.loc[session_df['rewardR']>0, 'respondTime'].values, legend = 'Reward in Right', color = 'b', plot_error = False, ax=ax)
        align_signal_to_events(curr_signal, signal['time_in_beh'], session_df.loc[session_df['rewardR']==0, 'respondTime'].values, legend = 'No Reward in Right', color = 'r', plot_error = False, ax=ax)
        ax.legend()
        ax.set_xlabel('Time from response (ms)')
        ax = fig.add_subplot(gs[1, 4])
        choices_mask = ~session_df['rewardL'].isna().values | ~session_df['rewardR'].isna().values
        choices = session_df.loc[choices_mask, 'rewardR'].isna().values.astype(int)
        change_trials = np.where(choices[:-1] != choices[1:])[0] + 1
        stay_trials = np.where(choices[:-1] == choices[1:])[0] + 1
        align_signal_to_events(curr_signal, signal['time_in_beh'], session_df.loc[choices_mask, 'CSon'].values[change_trials], legend = 'Change', color = 'b', plot_error = False, ax=ax)
        align_signal_to_events(curr_signal, signal['time_in_beh'], session_df.loc[choices_mask, 'CSon'].values[stay_trials], legend = 'Stay', color = 'r', plot_error = False, ax=ax)
        ax.legend()
        ax.set_xlabel('Time from go cue (ms)')
        plt.suptitle(f'{region} - {session}')
        plt.tight_layout()

        fig.savefig(os.path.join(session_dir['saveFigFolder'], f'{region}_FP_beh_analysis_{channel}.pdf'))

def plot_FP_beh_analysis_model(session, model, category, channels = ['G_tri-exp_mc', 'Iso_tri-exp_mc'], regions = None, focus = 'pe', num_bins = 4, pre = 2000, post = 2500):
    """
    Plot the photometry signal with the behavior analysis and the RL model
    """
    aligns = ['CSon', 'respondTime']
    formula = 'spikes ~ 1 + outcome + choice + Qchosen'
    session_dirs = parse_session_string(session)
    beh_session_data, licksL, licksR = load_session_df(session)
    s = beh_analysis_no_plot(session)
    params, model_name, _, no_session = get_stan_model_params_samps_only(session_dirs['aniName'], category, model, 2000, session_name=session, plot_flag=False)
    t = infer_model_var(session, params, model_name, bias_flag=1, rev_for_flag=0)
    signal,_ = get_FP_data(session)
    target = t[focus]
    # if num_bins is even number, separate equal number of cases with target is larger and smaller than 0
    if num_bins % 2 == 0:
        edges_neg = np.quantile(target[target<0], np.linspace(0, 1, num_bins//2+1))
        edges_pos = np.quantile(target[target>0], np.linspace(0, 1, num_bins//2+1))
        edges = np.concatenate([edges_neg[:-1], np.zeros((1)), edges_pos[1:]])
    else:
        edges = np.quantile(target, np.linspace(0, 1, num_bins+1))
    edges[0] = edges[0] - 0.01
    edges[-1] = edges[-1] + 0.01

    color_points = [[1, 0, 0], [1, 1, 1], [0, 0, 1]]
    colors = color_gradient(color_points, num_bins)
    cmap = plt.get_cmap('viridis')

    # prepare dataframe for linear regression
    Qsum = np.sum(t['Q'], 1)
    Qdiff = t['Q'][:,1] - t['Q'][:,0]
    Qchosen = t['Q'][:,1]
    Qchosen[s['allChoices'] == -1] = t['Q'][:,0][s['allChoices'] == -1]
    Qunchosen = t['Q'][:,0]
    Qunchosen[s['allChoices'] == -1] = t['Q'][:,1][s['allChoices'] == -1]
    QdiffC = Qchosen - Qunchosen
    trial_data = pd.DataFrame({'outcome': s['allRewardsBinary'], 'choice': s['allChoices']>0, 'Qchosen': Qchosen, 'svs':s['svs'], 'Qsum': Qsum, 'Qdiff': Qdiff, 'QdiffC': QdiffC})
    trial_data = pd.concat([trial_data.reset_index(drop=True), beh_session_data.loc[s['responseInds']].reset_index(drop=True)], axis = 1)

    if regions is None:
        regions = signal['G'].keys()
    for region in regions:
        for channel in channels:
            curr_signal = zscore(signal[channel][region])
            gs = GridSpec(2, 3*len(aligns), height_ratios=[1, 1], width_ratios=np.ones(3*len(aligns)))
            gs_lm = GridSpec(2, 2*len(aligns), height_ratios=[1, 1], width_ratios=[2, 1]*len(aligns))
            fig = plt.figure(figsize=(20, 10))
            for align_ind, align in enumerate(aligns):
                ax = fig.add_subplot(gs[0, 3*align_ind])
                align_signal_to_events(
                    curr_signal, 
                    signal['time_in_beh'], 
                    beh_session_data.loc[np.array(s['responseInds'])[s['allRewardsBinary']>0], align].values, 
                    legend = 'reward', 
                    color = 'b', 
                    plot_error = False, 
                    pre_event_time=pre, post_event_time=post, 
                    ax=ax);
                align_signal_to_events(
                    curr_signal, 
                    signal['time_in_beh'], 
                    beh_session_data.loc[np.array(s['responseInds'])[s['allRewardsBinary']==0], align].values, 
                    legend = 'no reward', 
                    color = 'r', 
                    plot_error = False,
                    pre_event_time=pre, post_event_time=post, 
                    ax=ax);
                ax.legend()
                ax.set_xlabel(f"Time from {align} (ms)")
                ax.set_ylabel("zscored dF/F")
                ax = fig.add_subplot(gs[0, 3*align_ind+1])
                for curr_bin in range(num_bins):
                    align_signal_to_events(
                        curr_signal, 
                        signal['time_in_beh'], 
                        beh_session_data.loc[
                            np.array(s['responseInds'])[
                                (target >= edges[curr_bin]) & (target < edges[curr_bin + 1])
                            ], 
                            align
                        ].values, 
                        legend=f'{curr_bin}, Mean {np.mean(target[(target >= edges[curr_bin]) & (target < edges[curr_bin + 1])]):.2f}', 
                        color=colors[curr_bin], 
                        plot_error=False, 
                        pre_event_time=pre, post_event_time=post, 
                        ax=ax
                    );
                ax.legend()
                ax.set_xlabel(f"Time from {align} (ms)")

                ax = fig.add_subplot(gs[0, 3*align_ind+2])
                align_signal_to_events(
                    curr_signal, 
                    signal['time_in_beh'], 
                    beh_session_data.loc[np.array(s['responseInds'])[s['svs']], align].values, 
                    legend = 'switch', 
                    color = 'r', 
                    plot_error = False, 
                    pre_event_time=pre, post_event_time=post, 
                    ax=ax);
                align_signal_to_events(
                    curr_signal,
                    signal['time_in_beh'],
                    beh_session_data.loc[np.array(s['responseInds'])[~s['svs']], align].values,
                    legend='stay',
                    color='b',
                    plot_error=False,
                    pre_event_time=pre, post_event_time=post,
                    ax=ax);
                
                ax.set_xlabel(f"Time from {align} (ms)")
                ax.set_ylabel("zscored dF/F")
                ax.legend()
                

                ax = fig.add_subplot(gs_lm[1, 2*align_ind])
                # glm
                aligned_matrix, _, time_bins, _ = align_signal_to_events(
                                                                curr_signal, 
                                                                signal['time_in_beh'], 
                                                                beh_session_data.loc[np.array(s['responseInds']), align].values, 
                                                                legend = 'reward', 
                                                                color = 'b', 
                                                                plot_error = False, 
                                                                pre_event_time=2000, post_event_time=3000,
                                                                step_size=200, window_size=1000);
                if  not np.isnan(aligned_matrix).all():
                    regressors, TvCurrU, PvCurrU, EvCurrU = fitSpikeModelG(trial_data, aligned_matrix, formula)
                    colors_lm = cmap(np.linspace(0, 1, len(regressors)))
                    TvCurrUSig = TvCurrU.copy()
                    TvCurrUSig[PvCurrU>=0.05] = np.nan
                    for regress in range(1, len(regressors)):
                        ax.plot(time_bins, TvCurrU[:, regress], lw = 1, color = colors_lm[regress,], label = regressors[regress])
                        ax.plot(time_bins, TvCurrUSig[:, regress], lw = 3, color = colors_lm[regress,])
                    ax.legend()
                    # contruct regressor matrix for glm
                    # run linear regression model in each column of aligned_matrix
                    ax.set_xlabel(f"Time from {align} (ms)")
                    ax.set_title('Tstats')
            # plt.tight_layout()
            fig.suptitle(f'{session} {region} {channel} quantile bins by {focus}')
            fig.savefig(os.path.join(session_dirs['saveFigFolder'], f'{session}_{region}_{channel}_quantile_bins_{focus}.pdf'))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
def correlate_nan(x, y, lag='full'):
    """Calculate correlation while ignoring NaNs."""
    if lag == 'full':
        lag = len(x) - 1
    corrs = np.full((lag + 1,), np.nan)
    for l in range(lag + 1):
        if l==0:
            valid_mask = ~np.isnan(x) & ~np.isnan(y)
            corrs[l] = np.corrcoef(x[valid_mask], y[valid_mask])[0, 1]
        else:
            valid_mask = ~np.isnan(x[:-l]) & ~np.isnan(y[l:])
            if np.any(valid_mask):
                corrs[l] = np.corrcoef(x[:-l][valid_mask], y[l:][valid_mask])[0, 1]
    return corrs

def plot_FP_peaks_behavior(session):
    bin_size  = 0.05
    time_len = 3
    lags = int(np.floor(time_len/bin_size))
    acf_lags = np.arange(0, lags+1) * bin_size
    session_dir = parse_session_string(session)
    session_df, licks_L, licks_R = load_session_df(session)
    licks_L, licks_R, fig = clean_up_licks(licks_L, licks_R, plot=False)
    parsed_licks_L, _ = parse_lick_trains(licks_L)
    parsed_licks_R, _ = parse_lick_trains(licks_R)
    signal, params = get_FP_data(session)
    time = signal['time_in_beh']
    peaks = peak_detect_FP(session)
    targets = list(signal['G'].keys())
    colors = ["blue", "white", "red"]
    custom_cmap = LinearSegmentedColormap.from_list("blue_white_red", colors)

    fig = plt.figure(figsize=(30, 15))
    gs = GridSpec(4, 2*len(targets), figure=fig, height_ratios=[0.5, 2, 2, 0.5])
    for ind_tar in range(len(targets)):
        target = targets[ind_tar]
        channel = 'G_tri-exp_mc'
        signal_curr = signal[channel][target]
        peaks_curr = peaks['peak_time'][target]
        if len(peaks_curr) < 10:
            continue
        # auto_corr
        acf_bins = np.arange(time[0], time[-1], bin_size)
        signal_curr_acf = signal_curr
        # for i in range(acf_bins.shape[0]-1):
        #     signal_curr_acf[i] = np.mean(signal_curr[(time >= acf_bins[i]) & (time < acf_bins[i+1])])
        # # calculate autocorrelation
        acf = correlate_nan(signal_curr_acf, signal_curr_acf, lag=lags)
        acf = acf / acf[0]  # normalize by the first value

        ax = fig.add_subplot(gs[0, ind_tar*2])
        ax.plot(acf_lags[1:], acf[1:], label=target, color='C0')
        ax.set_title(f'{target}')
        ax.set_xlabel('Lag (s)')
        ax.set_ylabel('Correlation')

        # peak shape
        pre_event = -1000
        post_event = 1000
        binSize = 50  # ms
        stepSize = 50  # ms
        align_time = peaks_curr
        slide_times = np.arange(pre_event, post_event + stepSize, stepSize)
        currArray = np.zeros((len(align_time), len(slide_times)))
        for i, event in enumerate(align_time):
            for j, t in enumerate(slide_times):
                t_curr = event + t
                ind = np.where((time >= t_curr-0.5*binSize) & (time < t_curr + 0.5*binSize))[0]
                if len(ind) > 0:
                    currArray[i, j] = np.mean(signal_curr[ind])

        in_trial = np.full(len(peaks_curr), 0)
        go_cues = session_df['CSon'].values

        # remove licks that are following a go cue within 1 second
        for i in range(len(go_cues)):
            ind = np.where((peaks_curr> go_cues[i]) & (peaks_curr < go_cues[i] + 1000))[0]
            in_trial[ind] = 1
        map_value = in_trial
        bins = np.linspace(np.min(map_value), np.max(map_value), 3)  # bins for histogram
        bins[0] = bins[0] - 0.0001
        bins[-1] = bins[-1] + 0.0001
        labels = ['out of trial', 'in trial']
        fig, ax = plot_rate(
                            currArray,
                            slide_times, 
                            map_value,
                            bins,
                            labels,
                            custom_cmap,
                            fig,
                            gs[0, ind_tar*2+1],
                            pre_event,
                            post_event,
                        )
        ax.set_xlabel('Time from peak (ms)')
        ax.set_title('Peak shape')


        pre_event = -1000
        post_event = 1500
        time_bin = 100  # ms


        # peak detection aligned to go cue
        lick_lat = session_df['respondTime'].values - session_df['CSon'].values 
        lick_lat = lick_lat[session_df['rewardL'].notna() | session_df['rewardR'].notna()]
        lick_ind = np.argsort(lick_lat)
        focus_values = np.full(len(session_df), np.nan)
        focus_values[session_df['rewardR'].notna()] = 1
        focus_values[session_df['rewardL'].notna()] = 0
        focus_values = focus_values[~np.isnan(focus_values)]

        align_time = session_df[session_df['rewardL'].notna() | session_df['rewardR'].notna()]['CSon'].values
        bin_counts = 2
        bins= np.linspace(np.min(focus_values), np.max(focus_values), bin_counts + 1)  # bins for histogram

        bins[0] = bins[0] - 0.0001
        bins[-1] = bins[-1] + 0.0001
        labels = ['L', 'R']
        fig, ax1, ax2 = plot_raster_rate(peaks_curr,
                                        align_time[lick_ind], 
                                        focus_values[lick_ind], # sorted by certain value
                                        bins,
                                        labels,
                                        custom_cmap,
                                        fig,
                                        gs[1, ind_tar*2:ind_tar*2+1],
                                        tb=pre_event,
                                        tf=post_event,
                                        time_bin=time_bin,
                                        )
        ax2.set_xlabel('Time from CSon (ms)')
        ax2.set_title('')
        ax1.set_title('Peaks aligned to CSon')
        # peak detection aligned to response

        align_time = session_df[session_df['rewardL'].notna() | session_df['rewardR'].notna()]['respondTime'].values
        bin_counts = 2
        bins= np.linspace(np.min(focus_values), np.max(focus_values), bin_counts + 1)  # bins for histogram

        bins[0] = bins[0] - 0.0001
        bins[-1] = bins[-1] + 0.0001
        labels = ['L', 'R']
        fig, ax1, ax2 = plot_raster_rate(peaks_curr,
                                        align_time[lick_ind], 
                                        focus_values[lick_ind], # sorted by certain value
                                        bins,
                                        labels,
                                        custom_cmap,
                                        fig,
                                        gs[1, ind_tar*2+1],
                                        tb=pre_event,
                                        tf=post_event,
                                        time_bin=time_bin,
                                        )
        ax2.set_xlabel('Time from choice (ms)')
        ax2.set_title('')
        ax1.set_title('Peaks aligned to choice')

        # aligned to lick starts
        pre_event = -1000
        post_event = 1500
        time_bin = 300  # ms

        L_licks = parsed_licks_L['train_starts']
        R_licks = parsed_licks_R['train_starts']
        go_cues = session_df['CSon'].values
        peaks_curr_iti = peaks_curr.copy()
        # remove licks that are following a go cue within 1 second
        for i in range(len(go_cues)):
            L_licks = np.delete(L_licks, np.where((L_licks > go_cues[i]) & (L_licks < go_cues[i] + 1000))[0])
            R_licks = np.delete(R_licks, np.where((R_licks > go_cues[i]) & (R_licks < go_cues[i] + 1000))[0])
            peaks_curr_iti = np.delete(peaks_curr_iti, np.where((peaks_curr_iti > go_cues[i]) & (peaks_curr_iti < go_cues[i] + 1000))[0])
        all_licks = np.concatenate((L_licks, R_licks))
        focus_values = np.concatenate((np.zeros(len(L_licks)), np.ones(len(R_licks))))


        align_time = all_licks
        bin_counts = 2
        bins= np.linspace(np.min(focus_values), np.max(focus_values), bin_counts + 1)  # bins for histogram

        bins[0] = bins[0] - 0.0001
        bins[-1] = bins[-1] + 0.0001
        labels = ['L', 'R']
        fig, ax1, ax2 = plot_raster_rate(peaks_curr_iti,
                                        align_time, 
                                        focus_values, # sorted by certain value
                                        bins,
                                        labels,
                                            custom_cmap,
                                            fig,
                                            gs[2, ind_tar*2],
                                            tb=pre_event,
                                            tf=post_event,
                                            time_bin=time_bin,
                                            )
        ax2.set_xlabel('Time from lick train start (ms)')
        ax2.set_title('')
        ax1.set_title('Peaks aligned to lick train start')


        pre_event = -1000
        post_event = 1500
        time_bin = 300  # ms
        align_time = session_df['CSon'].values
        focus_values = session_df['rewardR'].notna() | session_df['rewardL'].notna()
        bin_counts = 2
        bins= np.linspace(np.min(focus_values), np.max(focus_values), bin_counts + 1)  # bins for histogram
        bins[0] = bins[0] - 0.0001
        bins[-1] = bins[-1] + 0.0001
        labels = ['no response', 'response']
        fig, ax1, ax2 = plot_raster_rate(peaks_curr,
                                        align_time, 
                                        focus_values, # sorted by certain value
                                        bins,
                                        labels,
                                        custom_cmap,
                                        fig,
                                        gs[2, ind_tar*2+1],
                                        tb=pre_event,
                                        tf=post_event,
                                        time_bin=time_bin,
                                        )
        ax1.set_ylim(-0.5, len(align_time) + 0.5)
        ax2.set_xlabel('Time from CSon (ms)')
        ax2.set_title('')
        ax2.legend(loc='upper left')
        ax1.set_title('Peaks aligned to CSon')

    plt.suptitle(session)
    plt.tight_layout()
    plt.savefig(os.path.join(session_dir['saveFigFolder'], 'FP_peaks_behavior.pdf'))