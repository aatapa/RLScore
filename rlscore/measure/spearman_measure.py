from scipy.stats import spearmanr
import numpy as np
from measure_utilities import multitask
from rlscore.utilities import array_tools

def spearman_singletask(Y, P):
    return spearmanr(Y, P)[0]

def spearman_multitask(Y, P):
    return multitask(Y, P, spearman_singletask)

def spearman(Y, P):
    Y = array_tools.as_labelmatrix(Y)
    P = array_tools.as_labelmatrix(P)  
    return np.mean(spearman_multitask(Y, P))
spearman.iserror=False
