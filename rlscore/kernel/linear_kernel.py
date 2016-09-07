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

from scipy import sparse as sp

from rlscore.utilities import array_tools

class LinearKernel(object):
    """Linear kernel.
    
    k(xi,xj) = <xi , xj> + bias

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = [n_bvectors, n_features]
        Basis vectors
    bias : float, optional (default 1.0)
        Constant added to each kernel evaluation
        
    Attributes
    ----------
    train_X : {array-like, sparse matrix}, shape = [n_bvectors, n_features]
        Basis vectors
    bias : float
        Constant added to each kernel evaluation
    """

    def __init__(self, X, bias=1.0):
        X = array_tools.as_2d_array(X, True)
        self.train_X = X
        self.bias = bias

    
    def getKM(self, X):
        """Returns the kernel matrix between the basis vectors and X.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        
        Returns
        -------
        K : array, shape = [n_samples, n_bvectors]
            kernel matrix
        """
        X = array_tools.as_2d_array(X, True)
        test_X = X
        if sp.issparse(test_X):
            test_X = array_tools.spmat_resize(test_X, self.train_X.shape[1])
        else:
            test_X = array_tools.as_dense_matrix(test_X)
        train_X = self.train_X
        K = train_X * test_X.T
        K = array_tools.as_array(K)
        if self.bias != 0:
            K += self.bias
        return K.T

