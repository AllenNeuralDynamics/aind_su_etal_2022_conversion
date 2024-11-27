import numpy as np
import scipy.io as sio
from utils.basics.data_org import parse_session_string, curr_computer
from utils.behavior.session_utils import beh_analysis_no_plot_opmd


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

    os = beh_analysis_no_plot_opmd(session, simple_flag=1)
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
