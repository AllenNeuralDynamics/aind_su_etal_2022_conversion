from utils.behavior.qLearning_models import*
from utils.basics.data_org import curr_computer
from utils.behavior.session_utils import beh_analysis_no_plot
from scipy.io import loadmat
import numpy as np
import random
import os
import re
import matplotlib.pyplot as plt

def get_model_variables_dF(model_name, params, choice, outcome, laser=None):
    t = {}

    model_functions = {
        "5params": qLearning_model_5params,
        "5params_k_bias": qLearning_model_5params_k,
        # "5paramsLaserNegRPE": qLearningModel_5paramsLaserNegRPE,
        # "5paramsLaserNegRPERotation": qLearningModel_5paramsLaserNegRPERotation_simNoPlot,
    }
    

    # Call the appropriate function if it exists
    if model_name in model_functions:
        if "LaserNegRPE" in model_name:
            t["LH"], t["probChoice"], t["Q"], t["pe"], t["peChange"] = model_functions[model_name](params, choice, outcome, laser)
        elif "fiveParam_ph_bias" in model_name or "sixParam_ph_bias" in model_name:
            t["LH"], t["probChoice"], t["Q"], t["pe"], t["alpha"] = model_functions[model_name](params, choice, outcome)
        else:
            t["LH"], t["probChoice"], t["Q"], t["pe"] = model_functions[model_name](params, choice, outcome)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return t

def get_param_names_dF(model_name, bias_flag):
    """
    Get parameter names based on the model name.

    Parameters:
        model_name (str): The name of the model.
        bias_flag (bool): Whether to include 'bias' in the parameters.

    Returns:
        list: A list of parameter names.
    """

    model_params = {
        "fourParam": ['aN', 'aP', 'aF', 'beta'],
        "4params_bias_LaserNegRPE": ['a', 'aF', 'beta', 'diff'],
        "5params": ['aN', 'aP', 'aF', 'beta'],
        "5params_biForget": ['aN', 'aP', 'aF', 'beta'],
        "5paramsHmm": ['aN', 'aP', 'aF', 'betaE', 'betaT'],
        "5paramsLaserNegRPE": ['aN', 'aP', 'aF', 'beta', 'diff'],
        "5paramsLaserNegRPERotation": ['aN', 'aP', 'aF', 'beta', 'diff'],
        "5paramsLaserUpdate": ['aN', 'aP', 'aF', 'beta', 'diff'],
        "5paramsExpForgetHmm": ['aN', 'aP', 'aF', 'betaE', 'betaT'],
        "5paramsNoPriorForgetHmm": ['aN', 'aP', 'aF', 'betaE', 'betaT'],
        "5paramsNoPriorForget": ['aN', 'aP', 'aF', 'beta'],
        "5params_tF": ['aN', 'aP', 'tF', 'beta'],
        "5params_biaF": ['aN', 'aP', 'aF', 'beta'],
        "5params_expF": ['aN', 'aP', 'aF', 'beta'],
        "5params_k_bias": ['a', 'aF', 'beta', 'k'],
        "5params_k_bias_biForget": ['a', 'aF', 'beta', 'k'],
        "5params_k_bias_LaserNegRPE": ['a', 'aF', 'beta', 'k', 'diff'],
        "5params_k_bias_LaserNegRPE_expPrior": ['a', 'aF', 'beta', 'k', 'diff'],
        "5params_k_bias_LaserDisengage": ['a', 'aF', 'beta', 'k', 'diff'],
        "5params_k_bias_LaserDisengageScale": ['a', 'aF', 'beta', 'k', 'scale'],
        "5params_k_bias_LaserDisengageScale_expPrior": ['a', 'aF', 'beta', 'k', 'scale'],
        "5params_k_bias_LaserNegOnlyRPE": ['a', 'aF', 'beta', 'k', 'diff'],
        "5params_k_bias_LaserNegRPERotation": ['a', 'aF', 'beta', 'k', 'diff'],
        "5params_2LR_k_bias": ['aN', 'aP', 'beta', 'k'],
        "5params_k_biForget_bias": ['a', 'aF', 'beta', 'k'],
        "5params_kExp_bias": ['a', 'aF', 'aChoice', 'beta', 'k'],
        "5params_kExp_bias_LaserNegRPERotation": ['a', 'aF', 'aChoice', 'beta', 'k', 'diff'],
        "5params_inv": ['aN', 'aP', 'aF', 'beta'],
        "fiveParam_kappa": ['aN', 'aP', 'aF', 'beta', 'k'],
        "fiveParam_ph_bias": ['eta', 'kappa', 'aF', 'beta'],
        "sixParam_ph_bias": ['eta', 'kappaN', 'kappaP', 'aF', 'beta'],
        "sixParam_absPePe_scale_bias": ['aN', 'aP', 'aF', 'aPE', 'beta'],
        "sixParam_absPePe_scaleBoth_bias": ['aN', 'aP', 'aF', 'aPE', 'beta'],
        "sixParam_absPePeAN_bi_bias": ['aNmin', 'aP', 'aF', 'aPE', 'beta'],
        "sixParam_absPePeAN_bi_scale_bias": ['aNmin', 'aP', 'aF', 'aPE', 'beta'],
        "sixParam_absPePeAN_bi_bias_noF": ['aNmin', 'aNscale', 'aP', 'aPE', 'beta'],
        "sevenParam_absPePeAN_bi_bias": ['aNmin', 'aNscale', 'aP', 'aF', 'aPE', 'beta'],
        "sevenParam_absPePeAN_int_bias": ['aNmin', 'aP', 'aF', 'aPE', 'v', 'beta'],
        "sevenParam_absPePeAN_scale_int_bias": ['aNmin', 'aP', 'aF', 'aPE', 'v', 'beta'],
        "7params_absPePeAN_scale_int_bias_ord": ['aNmin', 'aP', 'aF', 'aPE', 'v', 'beta'],
        "sevenParam_absPePeAN_int_bias_ord": ['aNmin', 'aP', 'aF', 'aPE', 'v', 'beta'],
        "eightParam_absPePeAN_scale_int_bias": ['aNmin', 'aNscale', 'aP', 'aF', 'aPE', 'v', 'beta'],
        "sixParam_rTrace": ['a', 'aF', 'beta', 'v', 'w'],
        "sevenParam_rTrace_k": ['a', 'aF', 'beta', 'v', 'w', 'k'],
        "sixParam_ph_bias": ['eta', 'kappaN', 'kappaP', 'aF', 'beta'],

        # Delta models
        "delta_sevenParam_absPePeAN_scale_int_bias_ord": ['aNmin', 'aP', 'aF', 'aPE', 'v', 'beta', 'dPeBar'],

        # Bayesian models
        "fbm_tsPrior": ['a', 'b'],
        "fbm_softmax": ['a', 'b', 'beta'],
        "dbm_tsPrior": ['gamma', 'a', 'b'],
        "dbm_softmax": ['gamma', 'a', 'b', 'beta'],
        "dbm_softmax_bias": ['gamma', 'a', 'b', 'beta', 'bias'],
        "dbm_softmax_bias_probs": ['gamma', 'beta', 'bias'],
        "dbm_softmax_ab": ['gamma', 'ab', 'beta'],

        # VKF models
        "vkf": ['lambda', 'vInit', 'omega', 'beta'],
        "vkf_fixV": ['lambda', 'vInit', 'omega', 'beta'],
        "vkf_fixV_aF": ['lambda', 'vInit', 'omega', 'beta', 'aF'],
        "vkf_fixV_kappa": ['lambda', 'vInit', 'omega', 'beta', 'kappa']
    }

    # Get the parameter list for the given model name
    param_names = model_params.get(model_name, [])

    # Append 'bias' if bias_flag is enabled
    if bias_flag:
        param_names.append('bias')

    return param_names

def get_stan_model_params_samps_only(animal_name, category, model_name, num_samps, 
                                     bias_flag=1, session_params_flag=1, 
                                     session_name=None, plot_flag=0):
    """
    Load model parameters from Stan samples.

    Parameters:
        animal_name (str): Name of the animal.
        category (str): Category of the model.
        model_name (str): Name of the model.
        num_samps (int): Number of samples.
        bias_flag (int, optional): Default is 1.
        session_params_flag (int, optional): Default is 1.
        session_name (str, optional): Specific session name. Default is None.
        plot_flag (int, optional): Whether to plot histograms. Default is 0.

    Returns:
        params (np.array): Extracted parameter samples.
        model_name (str): Name of the model.
        ll (float or np.array): Log likelihood values.
        no_session (bool): Flag indicating if session was not found.
    """

    # Get system-specific root path
    root = curr_computer()  # Assumed function to get base path
    param_names = get_param_names_dF(model_name, bias_flag)  # Get parameter names

    # Construct sample file name
    if re.match(r'^[A-Za-z]', animal_name):
        samp_file = f"{animal_name}{category}_{model_name}"
    else:
        samp_file = f"m{animal_name}{category}_{model_name}"

    # Construct paths
    path = os.path.join(root, animal_name, f"{animal_name}sorted", "stan", 
                        "bernoulli", model_name, category)
    model_path = os.path.join(path, f"{samp_file}.mat")

    # Initialize variables
    no_session = False
    params = np.full((num_samps, len(param_names)), np.nan)
    ll = np.nan

    # Load .mat file
    try:
        mat_data = loadmat(model_path)
    except FileNotFoundError:
        print(f"Error: File {model_path} not found.")
        return params, model_name, ll, no_session

    # Extract session index if sessionParamsFlag is enabled
    if session_params_flag and session_name:
        day_list = mat_data.get('dayList', [])
        session_ind = next((i for i, day in enumerate(day_list) if session_name in str(day)), None)
        if session_ind is None:
            no_session = True
            return params, model_name, ll, no_session

    # Load samples from the mat file
    if samp_file in mat_data:
        samples = mat_data[samp_file]
    else:
        print(f"Error: '{samp_file}' not found in .mat file.")
        return params, model_name, ll, no_session

    # Select valid non-divergent samples
    divergent = np.array(np.squeeze(samples['divergent__'][0][0])) < 1
    valid_inds = np.where(divergent)[0]
    if len(valid_inds) < num_samps:
        print("Warning: Not enough valid samples available.")
        return params, model_name, ll, no_session
    inds = random.sample(list(valid_inds), num_samps)

    # Extract parameter samples
    for i, param in enumerate(param_names):
        if session_params_flag:
            tmp = samples[param][0][0][:, session_ind]  # Extract session-specific parameters
        else:
            if param != "bias":
                tmp = samples[f"mu_{param}"][0][0]  # Extract population-level parameters
            else:
                tmp = np.zeros(len(samples[0][0]['beta']))  # Bias is set to zero
        params[:, i] = tmp[inds]

    # Extract log-likelihood if session-specific parameters are used
    if session_params_flag:
        ll = samples["log_lik"][0][0][inds, session_ind]

    # Plot histograms if required
    if plot_flag:
        plt.figure(figsize=(len(param_names) * 3, 3))
        colors = plt.cm.cool(np.linspace(0, 1, len(param_names)))
        for i, param in enumerate(param_names):
            plt.subplot(1, len(param_names), i + 1)
            plt.hist(params[:, i], bins=25, color=colors[i], alpha=0.7, density=True)
            plt.title(param)
        plt.show()

    return params, model_name, ll, no_session

def infer_model_var(session, params, model_name, bias_flag=1, rev_for_flag=0, perturb=None):
    """
    Infer model variables based on the given session and parameters.
    
    Parameters:
        session (object): Session data.
        params (np.array): Model parameters.
        model_name (str): Name of the model.
        bias_flag (int, optional): Default is 1.
        rev_for_flag (int, optional): Default is 0.
        perturb (list, optional): Perturbation settings. Default is None.
    
    Returns:
        dict: Model variables with computed Q-values and likelihoods.
    """

    t = {}

    # Get session behavior
    o = beh_analysis_no_plot(session, simple_flag=1)
    outcome = np.abs(np.array(o["allRewards"]))
    choice = np.array(o["allChoices"])
    choice[choice < 0] = 0  # Convert negative (left) choices to 0
    ITI = o["timeBtwn"]
    
    tmp_struct = []
    
    # Process models
    for currS in range(params.shape[0]):
        if perturb is None:
            tmp = get_model_variables_dF(model_name, params[currS, :], choice, outcome)
        # else: # to be updated when add back laser stimulation analysis
        #     tmp = get_model_variables_laser_dF(model_name, params[currS, :], choice, outcome, o["laser"])
        
        tmp_struct.append(tmp)

    # Handle infinite values
    mdl_var_names = list(tmp_struct[0].keys())
    inf_inds = [
        params_ind for params_ind in range(params.shape[0]) 
        if any(np.any(np.isinf(tmp_struct[params_ind][var_name])) for var_name in mdl_var_names)
    ]
    valid_inds = np.setdiff1d(np.arange(params.shape[0]), inf_inds)
    tmp_struct = [tmp_struct[i] for i in valid_inds]
    t["params"] = params[valid_inds, :]

    for currV in mdl_var_names:
        tmp = np.stack([tmp_struct[currS][currV] for currS in range(len(tmp_struct))], axis = 0)
        t[currV] = np.nanmean(tmp, axis=0)

    return t

 

