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

