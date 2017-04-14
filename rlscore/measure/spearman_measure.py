#
# The MIT License (MIT)
#
# This file is part of RLScore 
#
# Copyright (c) 2014 - 2016 Tapio Pahikkala, Antti Airola
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

from scipy.stats import spearmanr
import numpy as np
from .measure_utilities import multitask
from rlscore.utilities import array_tools
from rlscore.measure.measure_utilities import UndefinedPerformance

def spearman_singletask(Y, P):
    return spearmanr(Y, P)[0]

def spearman_multitask(Y, P):
    return multitask(Y, P, spearman_singletask)

def spearman(Y, P):
    """Spearman correlation.
    
    
    Parameters
    ----------
    Y : {array-like}, shape = [n_samples] or [n_samples, n_labels]
        Correct labels
    P : {array-like}, shape = [n_samples] or [n_samples, n_labels]
        Predicted labels
    
    Returns
    -------
    correlation : float
        number between -1 and 1
    """
    Y = array_tools.as_2d_array(Y)
    P = array_tools.as_2d_array(P)
    if not Y.shape == P.shape:
        raise UndefinedPerformance("Y and P must be of same shape")
    return np.mean(spearman_multitask(Y, P))
spearman.iserror=False
