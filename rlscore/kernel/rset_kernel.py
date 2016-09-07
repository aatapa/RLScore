#
# The MIT License (MIT)
#
# This file is part of RLScore 
#
# Copyright (c) 2012 - 2016 Tapio Pahikkala, Antti Airola
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

import numpy.linalg as la
import numpy as np

class RsetKernel(object):
    '''
    This class is for testing reduced set approximation.
    '''
    
    def __init__(self, base_kernel, X, basis_features):
        """Default implementation uses the scipy sparse matrices for internal representation of the data."""
        self.base_kernel = base_kernel
        Krr = self.base_kernel.getKM(basis_features)
        K_r = self.base_kernel.getKM(X)
        invKrr = la.inv(Krr)
        self.predcache = np.dot(K_r, invKrr)
        self.train_X = X

    
    def getKM(self, test_X):
        Ktr = self.base_kernel.getKM(test_X)
        return np.dot(Ktr, self.predcache.T)
    
    