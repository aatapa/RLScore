#Disagreement error, a standard ranking measure.
import numpy as np

from rlscore.measure.measure_utilities import UndefinedPerformance
from rlscore.utilities import array_tools

def disagreement_singletask(Y, P):
    correct = Y
    predictions = P
    assert len(correct) == len(predictions)
    disagreement = 0.
    decisions = 0.
    for i in range(len(correct)):
        for j in range(len(correct)):
                if correct[i] > correct[j]:
                    decisions += 1.
                    if predictions[i] < predictions[j]:
                        disagreement += 1.
                    elif predictions[i] == predictions[j]:
                        disagreement += 0.5
    #Disagreement error is not defined for cases where there
    #are no disagreeing pairs
    if decisions == 0:
        raise UndefinedPerformance("No pairs, all the instances have the same label")
    else:
        disagreement /= decisions
    return disagreement

def disagreement_multitask(Y, P):
    perfs = []
    for i in range(Y.shape[1]):
        try:
            perfs.append(disagreement_singletask(Y[:,i], P[:,i]))
        except UndefinedPerformance, e:
            perfs.append(np.nan)
    return perfs

def disagreement(Y, P):
    """Disagreement error, also known as the pairwise ranking error.
    
    A performance measure for ranking problems. Computes the number
    of pairwise disagreements between the correct and predicted rankings.
    An O(n^2)-time implementation, can be slow for large problems (loglinear
    time implementation would be possible using search trees). For query-structured
    data, one would typically want to compute the disagreement separately for each query,
    and average.
    
    If 2-dimensional arrays are supplied as arguments, then disagreement
    is separately computed for each column, after which the disagreements
    are averaged.
    
    Parameters
    ----------
    Y: {array-like}, shape = [n_samples] or [n_samples, n_labels]
        Correct utility values, can be any real numbers
    P: {array-like}, shape = [n_samples] or [n_samples, n_labels]
        Predicted utility values, can be any real numbers. 
    
    Returns
    -------
    disagreement: float
        number between 0 and 1
    """
    Y = array_tools.as_labelmatrix(Y)
    P = array_tools.as_labelmatrix(P)
    perfs = disagreement_multitask(Y,P)
    perfs = np.array(perfs)
    perfs = perfs[np.invert(np.isnan(perfs))]
    if len(perfs) == 0:
        raise UndefinedPerformance("No pairs, all the instances have the same label")
    perf = np.mean(perfs)
    return perf
disagreement.iserror = True
