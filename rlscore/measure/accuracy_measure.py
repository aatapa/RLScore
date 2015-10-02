import numpy as np
from rlscore.utilities import array_tools

def accuracy_singletask(Y, P):
    assert Y.shape[0] == P.shape[0]
    vlen = float(Y.shape[0])
    perf = np.sum(np.sign(np.multiply(Y, P)) + 1.) / (2 * vlen)
    return perf

def accuracy_multitask(Y, P):
    Y = np.mat(Y)
    P = np.mat(P)
    vlen = float(Y.shape[0])
    performances = sum(np.sign(np.multiply(Y, P)) + 1., axis = 0) / (2 * vlen)
    performances = np.array(performances)[0]
    return performances

def accuracy(Y, P):
    """Binary classification accuracy.
    
    A performance measure for binary classification problems.
    Returns the fraction of correct class predictions. P[i]>0 is
    considered a positive class prediction and P[i]<0 negative.
    P[i]==0 is considered as classifier abstaining to make a decision,
    which incurs 0.5 errors (in contrast to 0 error for correct and 1
    error for incorrect prediction).
    
    If 2-dimensional arrays are supplied as arguments, then accuracy
    is separately computed for each column, after which the accuracies
    are averaged.
    
    Parameters
    ----------
    Y: {array-like}, shape = [n_samples] or [n_samples, n_labels]
        Correct labels, must belong to set {-1,1}
    P: {array-like}, shape = [n_samples] or [n_samples, n_labels]
        Predicted labels, can be any real numbers. 
    
    Returns
    -------
    accuracy: float
        number between 0 and 1
    """
    Y = array_tools.as_labelmatrix(Y)
    P = array_tools.as_labelmatrix(P)    
    return np.mean(accuracy_multitask(Y, P))
accuracy.iserror = False