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
from scipy.signal import find_peaks

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

def parse_lick_trains(licks, window_size = 1000, height = 2, min_dist = 2000, inter_train_interval = 1000, inter_lick_interval = 300, plot = False):
    """
    """
    licks = np.array(licks)
    # Check unit of the data, if in s, convert to ms
    if np.mean(np.diff(licks)) < 100:
        licks = np.round(licks * 1000)
    # Lick peak detection
    bins = np.arange(licks.min(), licks.max(), 1)
    time_binned = bins[:-1]
    licks_binned = np.histogram(licks, bins=bins)[0]
    licks_smoothed = np.convolve(licks_binned, np.ones(window_size)/(window_size/1000), mode='same')
    peaks, lick_peak_amplitudes = find_peaks(licks_smoothed, height = height, distance = min_dist)
    lick_peak_amplitudes = lick_peak_amplitudes['peak_heights']
    lick_peak_times = time_binned[peaks]
    # lick train detection
    inter_lick_interval_mask = np.diff(licks)
    inter_train_mask = inter_lick_interval_mask > inter_train_interval
    within_train_mask = inter_lick_interval_mask < inter_lick_interval
    pre_it_mask = np.concatenate([[True], inter_train_mask])
    post_it_mask = np.concatenate([inter_train_mask, [True]])
    pre_wt_mask = np.concatenate([[False], within_train_mask])
    post_wt_mask = np.concatenate([within_train_mask, [False]])
    train_starts_tmp = licks[pre_it_mask & post_wt_mask]
    train_ends_tmp = licks[pre_wt_mask & post_it_mask]
    if len(train_starts_tmp) > len(train_ends_tmp):
        train_starts_tmp = train_starts_tmp[:-1]
    train_starts = []
    train_ends = []
    # for every train_start, find the closest train_end that is larger than train_start
    for train_start in train_starts_tmp:
        train_end = train_ends_tmp[train_ends_tmp > train_start][0]
        train_starts.append(train_start)
        train_ends.append(train_end)
    
    fig = None
    if plot:
        fig, ax = plt.subplots(1, 1, figsize = (10, 3), sharex = True)
        ax.plot(time_binned, licks_smoothed, label = 'Lick rate')
        ax.plot(lick_peak_times, lick_peak_amplitudes, 'ro', label = 'Lick peak')
        ax.plot(time_binned, licks_binned, label = 'Lick count')
        ax.set_title('Lick rate')
        for start, end in zip(train_starts, train_ends):
            ax.fill_between([start, end], 0, 100, color = 'gray', alpha = 0.5)
        ax.legend()
        ax.set_xlim([licks.min(), licks.min() + 60*1000])
        ax.set_ylim([0, 20])
    parsed_licks = {'lick_peak_times': lick_peak_times, 'lick_peak_amplitudes': lick_peak_amplitudes, 'train_starts': train_starts, 'train_ends': train_ends}
    
    return parsed_licks, fig