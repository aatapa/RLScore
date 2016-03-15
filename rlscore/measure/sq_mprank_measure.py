import numpy as np

from rlscore.utilities import array_tools
from rlscore.measure.measure_utilities import UndefinedPerformance

def sqmprank_singletask(Y, P):
    correct = Y
    predictions = P
    correct = np.mat(correct)
    predictions = np.mat(predictions)
    vlen = correct.shape[0]
    diff = correct - predictions
    onevec = np.mat(np.ones((vlen, 1)))
    centereddiff = vlen * diff - onevec * (onevec.T * diff)
    sqerror = (centereddiff.T * diff)[0, 0] / ((len(correct) ** 2 - len(correct)) / 2)
    return sqerror

def sqmprank_multitask(Y, Y_predicted):  
    Y = np.mat(Y)
    Y_predicted = np.mat(Y_predicted)
    vlen = Y.shape[0]
    centeredsqerror = Y - Y_predicted
    onevec = np.mat(np.ones((vlen, 1)))
    tempvec = onevec.T * centeredsqerror
    np.multiply(vlen, centeredsqerror, centeredsqerror)
    np.subtract(centeredsqerror, tempvec, centeredsqerror)
    np.multiply(centeredsqerror, centeredsqerror, centeredsqerror)
    performances = np.mean(centeredsqerror, axis = 0) / ((vlen ** 2 - vlen) / 2)
    performances = np.array(performances)[0]
    return performances

def sqmprank(Y, P):
    """Squared magnitude preserving ranking error.
    
    A performance measure for ranking problems. Computes the sum of (Y[i]-Y[j]-P[i]+P[j])**2
    over all index pairs. normalized by the number of pairs. For query-structured data,
    one would typically want to compute the error separately for each query, and average.
    
    If 2-dimensional arrays are supplied as arguments, then error is separately computed for
    each column, after which the errors are averaged.
    
    Parameters
    ----------
    Y: {array-like}, shape = [n_samples] or [n_samples, n_labels]
        Correct utility values, can be any real numbers
    P: {array-like}, shape = [n_samples] or [n_samples, n_labels]
        Predicted utility values, can be any real numbers. 
    
    Returns
    -------
    error: float
    """
    Y = array_tools.as_2d_array(Y)
    P = array_tools.as_2d_array(P)
    if not Y.shape == P.shape:
        raise UndefinedPerformance("Y and P must be of same shape")
    return np.mean(sqmprank_multitask(Y, P))
sqmprank.iserror = True