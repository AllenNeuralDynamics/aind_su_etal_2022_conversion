import numpy as np
from scipy.special import expit  # Sigmoid function for logistic transformation

def qLearning_model_5params(start_values, choice, outcome):
    alphaNPE = start_values[0]
    alphaPPE = start_values[1]
    alphaForget = start_values[2]
    beta = start_values[3]
    bias = start_values[4]

    trials = len(choice)
    Q = np.zeros((trials, 2))
    pe = np.zeros(trials)

    # Call learning rule
    for t in range(trials - 1):
        if choice[t] == 1:  # Right choice
            Q[t + 1, 0] = alphaForget * Q[t, 0]
            pe[t] = outcome[t] - Q[t, 1]
            Q[t + 1, 1] = Q[t, 1] + (alphaNPE if pe[t] < 0 else alphaPPE) * pe[t]
        else:  # Left choice
            Q[t + 1, 1] = alphaForget * Q[t, 1]
            pe[t] = outcome[t] - Q[t, 0]
            Q[t + 1, 0] = Q[t, 0] + (alphaNPE if pe[t] < 0 else alphaPPE) * pe[t]

    # Compute last prediction error
    pe[-1] = outcome[-1] - Q[-1, 1] if choice[-1] == 1 else outcome[-1] - Q[-1, 0]

    # Softmax rule (logistic function)
    probChoice = expit(beta * (Q[:, 1] - Q[:, 0]) + bias)

    # Calculate likelihood
    LH = likelihood(choice, probChoice)

    # Compute chosen probabilities
    probChosen = np.where(choice == 0, 1 - probChoice, probChoice)

    return LH, probChosen, Q, pe

def qLearning_model_5params_k(start_values, choice, outcome):
    alpha = start_values[0]
    aF = start_values[1]
    beta = start_values[2]
    k = start_values[3]
    bias = start_values[4]

    trials = len(choice)
    Q = np.zeros((trials, 2))
    kChoice = np.zeros(trials)
    pe = np.zeros(trials)

    # Call learning rule
    for t in range(trials - 1):
        if choice[t] == 1:  # Right choice
            Q[t + 1, 0] = aF * Q[t, 0]
            pe[t] = outcome[t] - Q[t, 1]
            Q[t + 1, 1] = Q[t, 1] + alpha * pe[t]
            kChoice[t + 1] = k
        else:  # Left choice
            Q[t + 1, 1] = aF * Q[t, 1]
            pe[t] = outcome[t] - Q[t, 0]
            Q[t + 1, 0] = Q[t, 0] + alpha * pe[t]
            kChoice[t + 1] = -k

    # Compute last prediction error
    pe[-1] = outcome[-1] - Q[-1, 1] if choice[-1] == 1 else outcome[-1] - Q[-1, 0]

    # Softmax rule (logistic function)
    probChoice = expit(beta * (Q[:, 1] - Q[:, 0]) + kChoice + bias)

    # Calculate likelihood
    LH = likelihood(choice, probChoice)

    # Compute chosen probabilities
    probChosen = np.where(choice == 0, 1 - probChoice, probChoice)

    return LH, probChosen, Q, pe

def likelihood(choice, probChoice):
    """Computes likelihood given choices and probabilities."""
    return np.sum(np.log(probChoice[choice == 1])) + np.sum(np.log(1 - probChoice[choice == 0]))