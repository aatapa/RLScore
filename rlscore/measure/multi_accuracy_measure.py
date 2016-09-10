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
