import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.append('..') 
import platform
import os
from pathlib import Path
from utils.basics.data_org import curr_computer
import shutil
from utils.behavior.session_utils import load_session_df, parse_session_string
from utils.behavior.lick_analysis import clean_up_licks, parse_lick_trains
from itertools import chain
from matplotlib import pyplot as plt

def align_signal_to_events(signal, signal_time, event_times, pre_event_time=1000, post_event_time=2000, window_size=100, step_size=10, ax = None, legend = 'signal', color = 'b'):
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
    num_steps = (pre_event_time + post_event_time - window_size) // step_size + 1
    aligned_matrix = np.zeros((len(event_times), num_steps))
    time_bins = np.arange(-pre_event_time, post_event_time - window_size + step_size, step_size)

    for i, event_time in enumerate(event_times):
        start_time = event_time - pre_event_time
        end_time = event_time + post_event_time
        mask = (signal_time >= start_time-0.5*window_size) & (signal_time < end_time+0.5*window_size)
        aligned_signal = signal[mask]
        aligned_time = signal_time[mask] - event_time

        for j in range(num_steps):
            window_start = j * step_size - pre_event_time - 0.5*window_size
            window_end = window_start + window_size
            window_mask = (aligned_time >= window_start) & (aligned_time < window_end)
            aligned_matrix[i, j] = np.nanmean(aligned_signal[window_mask])

    mean_psth = np.nanmean(aligned_matrix, axis=0)
    std_psth = np.nanstd(aligned_matrix, axis=0)
    se_psth = std_psth / np.sqrt(aligned_matrix.shape[0])

    # Plotting the PSTH
    if ax is not None:
        ax.plot(time_bins, mean_psth, color=color, label=legend)
        ax.fill_between(time_bins, mean_psth - std_psth, mean_psth + std_psth, alpha=0.1, facecolor=color)
        ax.fill_between(time_bins, mean_psth - se_psth, mean_psth + se_psth, alpha=0.3, facecolor=color)
        ax.set_xlabel('Time (ms)')
    return aligned_matrix, mean_psth, time_bins, ax
