import numpy as np
from rlscore.measure.measure_utilities import UndefinedPerformance

def ova_accuracy(Y, P):
    """One-vs-all classification accuracy for multi-class problems.
    
    Computes the accuracy for a one-versus-all decomposed classification
    problem. Each column in Y and P correspond to one possible class label.
    On each row, exactly one column in Y is 1, all the rest must be -1. The
    prediction for the i:th example is computed by taking the argmax over
    the indices of row i in P. 
    
    Parameters
    ----------
    Y : {array-like}, shape = [n_samples] or [n_samples, n_classes]
        Correct labels, must belong to set {-1,1}, with exactly
        one 1 on each row.
    P : {array-like}, shape = [n_samples] or [n_samples, n_classes]
        Predicted labels, can be any real numbers.
    
    Returns
    -------
    accuracy : float
        number between 0 and 1
    """
    Y = np.array(Y)
    P = np.array(P)
    if not Y.shape == P.shape:
        raise UndefinedPerformance("Y and P must be of same shape")
    Y = np.argmax(Y, axis=1)
    P = np.argmax(P, axis=1)
    return np.mean(Y==P)
ova_accuracy.iserror = False
