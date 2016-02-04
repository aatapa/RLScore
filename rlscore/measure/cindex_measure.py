from numpy import array
import numpy as np
from rlscore.measure.measure_utilities import UndefinedPerformance
from rlscore.utilities import array_tools

def cindex_singletask(Y, P):
    Y = np.array(Y).T[0]
    P = np.array(P).T[0]
    correct = Y.astype(np.float64)
    predictions = P.astype(np.float64)
    assert len(correct) == len(predictions)
    C = array(correct).reshape(len(correct),)
    C.sort()
    pairs = 0
    c_ties = 0
    for i in range(1, len(C)):
        if C[i] != C[i-1]:
            c_ties = 0
        else:
            c_ties += 1
        #this example forms a pair with each previous example, that has a lower value
        pairs += i-c_ties
    if pairs == 0:
        raise UndefinedPerformance("No pairs, all the instances have the same output")
    correct = array(correct).reshape(correct.shape[0],)
    predictions = array(predictions).reshape(predictions.shape[0],)
    s = swapped.count_swapped(correct, predictions)
    disagreement = float(s)/float(pairs)
    return 1. - disagreement

def cindex_singletask_SLOW(Y, P):
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
        raise UndefinedPerformance("No pairs, all the instances have the same  output")
    else:
        disagreement /= decisions
    return 1. - disagreement

try:
    from rlscore.utilities import swapped
except Exception, e:
    print(e)
    print('Warning: could not import the fast cython implementation of the concordance index measure. Using a slow python-based one instead.')
    cindex_singletask = cindex_singletask_SLOW

def cindex_multitask(Y, P):
    perfs = []
    for i in range(Y.shape[1]):
        try:
            perfs.append(cindex_singletask(Y[:,i], P[:,i]))
        except UndefinedPerformance:
            perfs.append(np.nan)
    return perfs

def cindex(Y, P):
    """Concordance, aka pairwise ranking accuracy. Computes the
    relative fraction of concordant pairs, that is, Y[i] > Y[j]
    and P[i] > P[j] (ties with P[i]=P[j] are assumed to be broken
    randomly). Equivalent to area under ROC curve, if Y[i] belong
    to {-1, 1}. An O(n*log(n)) implementation, based on order
    statistic tree computations.
    
    Parameters
    ----------
    Y: {array-like}, shape = [n_samples] or [n_samples, n_labels]
        Correct labels, can be any real numbers. 
    P: {array-like}, shape = [n_samples] or [n_samples, n_labels]
        Predicted labels, can be any real numbers. 
    
    Returns
    -------
    concordance index: float
        number between 0 and 1, around 0.5 means random performance
    """
    Y = array_tools.as_labelmatrix(Y)
    P = array_tools.as_labelmatrix(P)
    perfs = cindex_multitask(Y,P)
    perfs = np.array(perfs)
    perfs = perfs[np.invert(np.isnan(perfs))]
    if len(perfs) == 0:
        raise UndefinedPerformance("No pairs, all the instances have the same output")
    return np.mean(perfs)
cindex.iserror = False

