#
# The MIT License (MIT)
#
# This file is part of RLScore 
#
# Copyright (c) 2016 Tapio Pahikkala, Antti Airola
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

def to_one_vs_all(Y, startzero=True):
    #maps vector of class labels from range {0,..,classcount-1} if startzero=True,
    #or range {1,...,classcount} if startzero=False to matrix,
    #of size ssize x classes
    Y = np.array(Y, dtype=int)
    if not startzero:
        Y = Y - 1
    Y_ova = -1. * np.ones((Y.shape[0], np.max(Y+1)))
    for i in range(len(Y)):
        Y_ova[i, Y[i]] = 1
    return Y_ova

def from_one_vs_all(Y, startzero=True):
    #maps one-vs-all encoding or predictions to classes ranging {0...classcount-1}
    #if fromzero=True, or {1...classcount} if startzero=False
    if startzero:
        return np.argmax(Y, axis=1)
    else:
        return np.argmax(Y, axis=1)+1
