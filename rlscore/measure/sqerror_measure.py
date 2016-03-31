from numpy import multiply, mean
import numpy as np

from rlscore.utilities import array_tools
from rlscore.measure.measure_utilities import UndefinedPerformance

def sqerror_singletask(correct, predictions):
    correct = np.mat(correct)
    predictions = np.mat(predictions)
    diff = correct - predictions
    sqerror = np.dot(diff.T, diff)[0,0]
    sqerror /= correct.shape[0]
    return sqerror
    
def sqerror_multitask(Y, Y_predicted):  
    Y = np.mat(Y)
    Y_predicted = np.mat(Y_predicted)
    sqerror = Y - Y_predicted
    multiply(sqerror, sqerror, sqerror)
    performances = mean(sqerror, axis = 0)
    performances = np.array(performances)[0]
    return performances

def sqerror(Y, P):
    """Mean squared error.
    
    A performance measure for regression problems. Computes the sum of (Y[i]-P[i])**2
    over all index pairs, normalized by the number of instances.
    
    If 2-dimensional arrays are supplied as arguments, then error is separately computed for
    each column, after which the errors are averaged.
    
    Parameters
    ----------
    Y : {array-like}, shape = [n_samples] or [n_samples, n_tasks]
        Correct utility values, can be any real numbers
    P : {array-like}, shape = [n_samples] or [n_samples, n_tasks]
        Predicted utility values, can be any real numbers. 
    
    Returns
    -------
    error : float
    """
    Y = array_tools.as_2d_array(Y)
    P = array_tools.as_2d_array(P)
    if not Y.shape == P.shape:
        raise UndefinedPerformance("Y and P must be of same shape")
    return np.mean(sqerror_multitask(Y,P))
sqerror.iserror = True

