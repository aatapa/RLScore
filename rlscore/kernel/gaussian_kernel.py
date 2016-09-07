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
from numpy import mat
from numpy import float64
from scipy import sparse as sp
from rlscore.utilities import array_tools


class GaussianKernel(object):
    """Gaussian (RBF) kernel.
    
    k(xi,xj) = e^(-gamma*<xi-xj,xi-xj>)

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = [n_bvectors, n_features]
        Basis vectors
    gamma : float, optional (default 1.0)
        Kernel width
        
    Attributes
    ----------
    train_X : {array-like, sparse matrix}, shape = [n_bvectors, n_features]
        Basis vectors
    gamma : float
        Kernel width
        
    """
      
    def __init__(self, X, gamma=1.0):
        X = array_tools.as_2d_array(X, True)
        if gamma <= 0.:
            raise Exception('ERROR: nonpositive kernel parameter for Gaussian kernel\n')
        self.train_X = X
        if sp.issparse(self.train_X):
            self.train_norms = ((self.train_X.T.multiply(self.train_X.T)).sum(axis=0)).T
        else:
            self.train_norms = np.mat((np.multiply(self.train_X.T, self.train_X.T).sum(axis=0))).T  
        self.gamma = gamma
            

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
        gamma = self.gamma
        m = self.train_X.shape[0]
        n = test_X.shape[0]
        #The Gaussian kernel matrix is constructed from a linear kernel matrix
        linkm = self.train_X * test_X.T
        linkm = array_tools.as_dense_matrix(linkm)
        if sp.issparse(test_X):
            test_norms = ((test_X.T.multiply(test_X.T)).sum(axis=0)).T
        else:
            test_norms = (np.multiply(test_X.T, test_X.T).sum(axis=0)).T
        K = mat(np.ones((m, 1), dtype = float64)) * test_norms.T
        K = K + self.train_norms * mat(np.ones((1, n), dtype = float64))
        K = K - 2 * linkm
        K = - gamma * K
        K = np.exp(K)
        return K.A.T


