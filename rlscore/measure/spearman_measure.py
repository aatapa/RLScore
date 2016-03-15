from scipy.stats import spearmanr
import numpy as np
from measure_utilities import multitask
from rlscore.utilities import array_tools
from rlscore.measure.measure_utilities import UndefinedPerformance

def spearman_singletask(Y, P):
    return spearmanr(Y, P)[0]

def spearman_multitask(Y, P):
    return multitask(Y, P, spearman_singletask)

def spearman(Y, P):
    """Spearman correlation.
    
    
    Parameters
    ----------
    Y: {array-like}, shape = [n_samples] or [n_samples, n_labels]
        Correct labels
    P: {array-like}, shape = [n_samples] or [n_samples, n_labels]
        Predicted labels
    
    Returns
    -------
    correlation: float
        number between -1 and 1
    """
    Y = array_tools.as_2d_array(Y)
    P = array_tools.as_2d_array(P)
    if not Y.shape == P.shape:
        raise UndefinedPerformance("Y and P must be of same shape")
    return np.mean(spearman_multitask(Y, P))
spearman.iserror=False
