#
# The MIT License (MIT)
#
# This file is part of RLScore 
#
# Copyright (c) 2008 - 2016 Tapio Pahikkala, Antti Airola
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

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
    Y : {array-like}, shape = [n_samples] or [n_samples, n_labels]
        Correct utility values, can be any real numbers
    P : {array-like}, shape = [n_samples] or [n_samples, n_labels]
        Predicted utility values, can be any real numbers. 
    
    Returns
    -------
    error : float
    """
    Y = array_tools.as_2d_array(Y)
    P = array_tools.as_2d_array(P)
    if not Y.shape == P.shape:
        raise UndefinedPerformance("Y and P must be of same shape")
    return np.mean(sqmprank_multitask(Y, P))
sqmprank.iserror = True