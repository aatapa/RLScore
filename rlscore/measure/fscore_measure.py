#
# The MIT License (MIT)
#
# This file is part of RLScore 
#
# Copyright (c) 2010 - 2016 Tapio Pahikkala, Antti Airola
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

from .measure_utilities import multitask
from rlscore.utilities import array_tools
from rlscore.measure.measure_utilities import UndefinedPerformance

def fscore_singletask(Y, P):
    correct = Y
    predictions = P
    if not np.all((Y==1) + (Y==-1)):
        raise UndefinedPerformance("fscore accepts as Y-values only 1 and -1")
    assert len(correct) == len(predictions)
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(correct)):
        if correct[i] == 1:
            if predictions[i] > 0.:
                TP += 1
            else:
                FN += 1
        elif correct[i] == -1:
            if predictions[i] > 0.:
                FP += 1
        else:
            assert False
    P = float(TP)/(TP+FP)
    R = float(TP)/(TP+FN)
    F = 2.*(P*R)/(P+R)
    return F

def fscore_multitask(Y, P):
    return multitask(Y, P, fscore_singletask)

def fscore(Y, P):
    """F1-Score.
    
    A performance measure for binary classification problems.
    F1 = 2*(Precision*Recall)/(Precision+Recall)
    
    If 2-dimensional arrays are supplied as arguments, then macro-averaged
    F-score is computed over the columns.
    
    Parameters
    ----------
    Y : {array-like}, shape = [n_samples] or [n_samples, n_labels]
        Correct labels, must belong to set {-1,1}
    P : {array-like}, shape = [n_samples] or [n_samples, n_labels]
        Predicted labels, can be any real numbers. P[i]>0 is treated
        as a positive, and P[i]<=0 as a negative class prediction.
    
    Returns
    -------
    fscore : float
        number between 0 and 1
    """
    Y = array_tools.as_2d_array(Y)
    P = array_tools.as_2d_array(P)
    if not Y.shape == P.shape:
        raise UndefinedPerformance("Y and P must be of same shape")
    return np.mean(fscore_multitask(Y,P))
fscore.iserror = False
