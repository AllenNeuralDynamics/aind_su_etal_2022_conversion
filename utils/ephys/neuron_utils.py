import numpy as np
import scipy.io as sio
from utils.basics.data_org import parse_session_string, curr_computer
from utils.behavior.session_utils import beh_analysis_no_plot, load_df_from_mat
from scipy.io import loadmat
import os
import pandas as pd

def load_neuron_df(session):
    path_data = parse_session_string(session)
    file_path = os.path.join(path_data["sortedFolder"], f"{session}_sessionData_nL.mat")
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return None, None, None, None
    session_beh_df, licks_L, licks_R = load_df_from_mat(file_path)
    neuron_df  = load_neurons_from_mat(file_path)

    return session_beh_df, licks_L, licks_R, neuron_df


def load_neurons_from_mat(file_path):
    mat_data = loadmat(file_path)
    # Access the 'beh' struct
    if 'sessionData' in mat_data:
        beh = mat_data['sessionData']
    else:
        print("No 'sessionData' found in the .mat file.")
        return None
    beh_dict = {field: beh[field].squeeze() for field in beh.dtype.names}

    # Create a DataFrame from the dictionary
    beh_df = pd.DataFrame(beh_dict)
    # get a list of the neuron names: key names in the df that include 'TT'
    unit_names = [col for col in beh_df .columns if 'TT' in col]
    # get all spiketimes for each neuron
    all_spike_times = beh_df['allSpikes'].to_list()[:len(unit_names)]
    # all_spike_times = [spiketimes[0] for spiketimes in beh_df['allSpikes'].to_list() if spiketimes.shape[1] > 0 and spiketimes.shape[0] > 0]   
    neuron_df = pd.DataFrame({'unit': unit_names, 'spike_times': all_spike_times})    
    return neuron_df

def get_unit_mat_choice(session, unit, tb, tf, step_size, bin_size):
    # Get root path and separator
    root, sep = curr_computer()

    # Initialize outputs
    cell_choice = []
    mat_choice = []
    mat_choice_slide = []
    slide_time = []

    # Define time windows
    time = np.arange(-1000 * tb, 1000 * tf + 1)  # Inclusive range
    mid_points = np.arange(
        0.5 * bin_size + 1, len(time) - 0.5 * bin_size + 1, step_size
    )
    slide_time = mid_points - tb * 1000

    # Paths
    pd = parse_session_string(session, root, sep)
    neuralynx_data_path = f"{pd['sortedFolder']}{session}_sessionData_nL.mat"
    session_data = sio.loadmat(neuralynx_data_path)["sessionData"]

    os = beh_analysis_no_plot(session, simple_flag=1)
    spike_fields = list(session_data.dtype.names)
    clust = [i for i, field in enumerate(spike_fields) if unit in field]

    if len(os["behSessionData"]) != len(session_data):
        print(f"{session} realign")
        return cell_choice, mat_choice, mat_choice_slide, slide_time

    # Cell
    all_trial_spike_choice = []
    for k in range(len(os["responseInds"])):
        if os["responseInds"][k] == 1:
            prev_trial_spike = []
        else:
            prev_trial_spike_ind = session_data[os["responseInds"][k] - 1][
                spike_fields[clust[0]]
            ] > (session_data[os["responseInds"][k]]["respondTime"] - tb * 1000)
            prev_trial_spike = (
                session_data[os["responseInds"][k] - 1][spike_fields[clust[0]]][
                    prev_trial_spike_ind
                ]
                - session_data[os["responseInds"][k]]["respondTime"]
            )

        curr_trial_spike_ind = (
            session_data[os["responseInds"][k]][spike_fields[clust[0]]]
            < session_data[os["responseInds"][k]]["respondTime"] + tf * 1000
        ) & (
            session_data[os["responseInds"][k]][spike_fields[clust[0]]]
            > session_data[os["responseInds"][k]]["respondTime"] - tb * 1000
        )
        curr_trial_spike = (
            session_data[os["responseInds"][k]][spike_fields[clust[0]]][
                curr_trial_spike_ind
            ]
            - session_data[os["responseInds"][k]]["respondTime"]
        )

        all_trial_spike_choice.append(
            np.concatenate([prev_trial_spike, curr_trial_spike])
        )

    all_trial_spike_choice = [
        spike if len(spike) > 0 else np.zeros(0) for spike in all_trial_spike_choice
    ]
    cell_choice = all_trial_spike_choice

    # Mat
    trial_dur_diff = [
        (data["trialEnd"] - (data["rewardTime"] - os["rwdDelay"])) - tf * 1000
        for data in session_data
    ]
    trial_dur_diff[-1] = 0
    all_trial_spike_matx_choice = np.zeros((len(os["responseInds"]), len(time)))

    for j, spikes in enumerate(all_trial_spike_choice):
        temp_spike = spikes + tb * 1000
        temp_spike[temp_spike == 0] = 1
        all_trial_spike_matx_choice[j, temp_spike.astype(int)] = 1
        if trial_dur_diff[j] < 0:
            all_trial_spike_matx_choice[
                j,
                np.isnan(
                    all_trial_spike_matx_choice[
                        j, : len(all_trial_spike_matx_choice[j]) + trial_dur_diff[j]
                    ]
                ),
            ] = 0
        else:
            all_trial_spike_matx_choice[
                j, np.isnan(all_trial_spike_matx_choice[j, :])
            ] = 0

    mat_choice = all_trial_spike_matx_choice

    # Slide window
    all_trial_spike_matx_slide = np.zeros((len(os["responseInds"]), len(mid_points)))
    for w, mid in enumerate(mid_points):
        window = slice(int(mid - 0.5 * bin_size), int(mid + 0.5 * bin_size))
        all_trial_spike_matx_slide[:, w] = (
            np.nansum(all_trial_spike_matx_choice[:, window], axis=1) * 1000 / bin_size
        )

    mat_choice_slide = all_trial_spike_matx_slide

    return cell_choice, mat_choice, mat_choice_slide, slide_time


def get_unit_mat_cue(session, unit, tb, tf, step_size, bin_size):
    # Define time range and sliding window midpoints
    time = np.arange(-1000 * tb, 1000 * tf + 1)  # Inclusive range
    mid_points = np.arange(
        0.5 * bin_size + 1, len(time) - 0.5 * bin_size + 1, step_size
    )
    slide_time = mid_points - tb * 1000

    # Paths
    root, sep = curr_computer()
    pd = parse_session_string(session, root, sep)
    neuralynx_data_path = f"{pd['sortedFolder']}{session}_sessionData_nL.mat"
    session_data = sio.loadmat(neuralynx_data_path)["sessionData"]

    # Find the cluster associated with the unit
    spike_fields = session_data.dtype.names
    clust = [i for i, field in enumerate(spike_fields) if unit in field]

    # Initialize data storage
    all_trial_spike_choice = []

    # Process spikes for each trial
    for k in range(len(session_data)):
        if k == 0:  # No previous trial
            prev_trial_spike = np.array([])
        else:
            prev_trial_spike_ind = session_data[k - 1][spike_fields[clust[0]]] > (
                session_data[k]["CSon"] - tb * 1000
            )
            prev_trial_spike = (
                session_data[k - 1][spike_fields[clust[0]]][prev_trial_spike_ind]
                - session_data[k]["CSon"]
            )

        curr_trial_spike_ind = (
            session_data[k][spike_fields[clust[0]]]
            < session_data[k]["CSon"] + tf * 1000
        ) & (
            session_data[k][spike_fields[clust[0]]]
            > session_data[k]["CSon"] - tb * 1000
        )
        curr_trial_spike = (
            session_data[k][spike_fields[clust[0]]][curr_trial_spike_ind]
            - session_data[k]["CSon"]
        )

        all_trial_spike_choice.append(
            np.concatenate([prev_trial_spike, curr_trial_spike])
        )

    # Replace empty entries with zero arrays
    all_trial_spike_choice = [
        spike if spike.size > 0 else np.zeros(0) for spike in all_trial_spike_choice
    ]
    cell_cue = all_trial_spike_choice

    # Compute trial duration differences
    trial_dur_diff = [
        (data["trialEnd"] - data["CSon"]) - tf * 1000 for data in session_data
    ]
    trial_dur_diff[-1] = 0  # Last trial adjustment

    # Create spike matrix for each trial
    all_trial_spike_matx_choice = np.zeros((len(session_data), len(time)))

    for j, spikes in enumerate(all_trial_spike_choice):
        temp_spike = spikes + tb * 1000
        temp_spike[temp_spike == 0] = 1  # Prevent zero indexing issues
        all_trial_spike_matx_choice[j, temp_spike.astype(int)] = 1
        if trial_dur_diff[j] < 0:
            all_trial_spike_matx_choice[
                j, int(len(all_trial_spike_matx_choice[j]) + trial_dur_diff[j]) :
            ] = 0
        else:
            all_trial_spike_matx_choice[
                j, np.isnan(all_trial_spike_matx_choice[j, :])
            ] = 0

    mat_cue = all_trial_spike_matx_choice

    # Create sliding window matrix
    all_trial_spike_matx_slide = np.zeros((len(session_data), len(mid_points)))

    for w, mid in enumerate(mid_points):
        window = slice(int(mid - 0.5 * bin_size), int(mid + 0.5 * bin_size))
        all_trial_spike_matx_slide[:, w] = (
            np.nansum(all_trial_spike_matx_choice[:, window], axis=1) * 1000 / bin_size
        )

    mat_cue_slide = all_trial_spike_matx_slide

    return cell_cue, mat_cue, mat_cue_slide, slide_time

def isi_violations(spike_train, total_duration_s, isi_threshold_s=0.0015, min_isi_s=0):
    """
    Calculate Inter-Spike Interval (ISI) violations.

    See compute_isi_violations for additional documentation

    Parameters
    ----------
    spike_trains : list of np.ndarrays
        The spike times for each recording segment for one unit, in seconds.
    total_duration_s : float
        The total duration of the recording (in seconds).
    isi_threshold_s : float, default: 0.0015
        Threshold for classifying adjacent spikes as an ISI violation, in seconds.
        This is the biophysical refractory period.
    min_isi_s : float, default: 0
        Minimum possible inter-spike interval, in seconds.
        This is the artificial refractory period enforced
        by the data acquisition system or post-processing algorithms.

    Returns
    -------
    isi_violations_ratio : float
        The isi violation ratio described in [1].
    isi_violations_rate : float
        Rate of contaminating spikes as a fraction of overall rate.
        Higher values indicate more contamination.
    isi_violation_count : int
        Number of violations.
    """

    num_violations = 0
    num_spikes = 0

    isi_violations_ratio = np.float64(np.nan)
    isi_violations_rate = np.float64(np.nan)
    isi_violations_count = np.float64(np.nan)
    isi_violations_percentile = np.float64(np.nan)

    isis = np.diff(spike_train)
    num_spikes = len(spike_train)
    num_violations = np.sum(isis < isi_threshold_s)

    violation_time = 2 * num_spikes * (isi_threshold_s - min_isi_s)

    if num_spikes > 0:
        total_rate = num_spikes / total_duration_s
        violation_rate = num_violations / violation_time
        isi_violations_ratio = violation_rate / total_rate
        isi_violations_rate = num_violations / total_duration_s
        isi_violations_count = num_violations
        isi_violations_percentile = isi_violations_count / num_spikes

    return isi_violations_ratio, isi_violations_rate, isi_violations_count, isi_violations_percentile