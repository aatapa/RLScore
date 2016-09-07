#
# The MIT License (MIT)
#
# This file is part of RLScore 
#
# Copyright (c) 2015 - 2016 Tapio Pahikkala, Antti Airola
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
from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix


from rlscore.utilities import adapter
from rlscore.utilities import linalg
from rlscore.predictor import PredictorInterface

class PPRankRLS(PredictorInterface):
    """Regularized least-squares ranking (RankRLS) with pairwise preferences

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Data matrix
        
    pairs_start_inds : {array-like}, shape = [n_preferences]
        pairwise preferences: pairs_start_inds[i] > pairs_end_inds[i]
        
    pairs_end_inds : {array-like}, shape = [n_preferences]
        pairwise preferences: pairs_start_inds[i] > pairs_end_inds[i]
        
    regparam : float, optional
        regularization parameter, regparam > 0 (default=1.0)
        
    kernel : {'LinearKernel', 'GaussianKernel', 'PolynomialKernel', 'PrecomputedKernel', ...}
        kernel function name, imported dynamically from rlscore.kernel
        
    basis_vectors : {array-like, sparse matrix}, shape = [n_bvectors, n_features], optional
        basis vectors (typically a randomly chosen subset of the training data)
        
    Other Parameters
    ----------------
    bias : float, optional
        LinearKernel: the model is w*x + bias*w0, (default=1.0)
        
    gamma : float, optional
        GaussianKernel: k(xi,xj) = e^(-gamma*<xi-xj,xi-xj>) (default=1.0)
        PolynomialKernel: k(xi,xj) = (gamma * <xi, xj> + coef0)**degree (default=1.0)
               
    coef0 : float, optional
        PolynomialKernel: k(xi,xj) = (gamma * <xi, xj> + coef0)**degree (default=0.)
        
    degree : int, optional
        PolynomialKernel: k(xi,xj) = (gamma * <xi, xj> + coef0)**degree (default=2)
        
    Attributes
    -----------
    predictor : {LinearPredictor, KernelPredictor}
        trained predictor
                  
    Notes
    -----
    
    Computational complexity of training:
    m = n_samples, d = n_features, p = n_preferences, b = n_bvectors
    
    O(m^3 + dm^2 + p): basic case
    
    O(dm^2 + p): Linear Kernel, d < m
    
    O(bm^2 + p): Sparse approximation with basis vectors 
     
    RankRLS algorithm was generalized in [1] to learning directly from pairwise preferences.
    
    References
    ----------
    [1] Tapio Pahikkala, Evgeni Tsivtsivadze, Antti Airola, Jouni Jarvinen, and Jorma Boberg.
    An efficient algorithm for learning to rank from preference graphs.
    Machine Learning, 75(1):129-165, 2009.
    """


    def __init__(self, X, pairs_start_inds, pairs_end_inds, regparam = 1.0, kernel='LinearKernel', basis_vectors = None, **kwargs):
        
        kwargs['kernel'] =  kernel
        kwargs['X'] = X
        if basis_vectors is not None:
            kwargs["basis_vectors"] = basis_vectors
        self.regparam = regparam
        self.pairs = np.vstack([pairs_start_inds, pairs_end_inds]).T
        self.svdad = adapter.createSVDAdapter(**kwargs)
        self.svals = np.mat(self.svdad.svals)
        self.svecs = self.svdad.rsvecs
        self.results = {}
        self.X = csc_matrix(X)
        self.bias = 0.
        self.results = {}
        self.solve(regparam)
    
    
    def solve(self, regparam):
        """Re-trains RankRLS for the given regparam.
               
        Parameters
        ----------
        regparam : float, optional
            regularization parameter, regparam > 0 (default=1.0)
            
        Notes
        -----   
        """
        size = self.svecs.shape[0]
        
        if not hasattr(self, "multipleright"):
            vals = np.concatenate([np.ones((self.pairs.shape[0]), dtype=np.float64), -np.ones((self.pairs.shape[0]), dtype = np.float64)])
            row = np.concatenate([np.arange(self.pairs.shape[0]), np.arange(self.pairs.shape[0])])
            col = np.concatenate([self.pairs[:, 0], self.pairs[:, 1]])
            coo = coo_matrix((vals, (row, col)), shape = (self.pairs.shape[0], size))
            self.L = (coo.T * coo)#.todense()
            
            #Eigenvalues of the kernel matrix
            evals = np.multiply(self.svals, self.svals)
            
            #Temporary variables
            ssvecs = np.multiply(self.svecs, self.svals)
            
            #These are cached for later use in solve and computeHO functions
            ssvecsTLssvecs = ssvecs.T * self.L * ssvecs
            LRsvals, LRevecs = linalg.eig_psd(ssvecsTLssvecs)
            LRsvals = np.mat(LRsvals)
            LRevals = np.multiply(LRsvals, LRsvals)
            LY = coo.T * np.mat(np.ones((self.pairs.shape[0], 1)))
            self.multipleright = LRevecs.T * (ssvecs.T * LY)
            self.multipleleft = ssvecs * LRevecs
            self.LRevals = LRevals
            self.LRevecs = LRevecs
        
        
        self.regparam = regparam
        
        #Compute the eigenvalues determined by the given regularization parameter
        self.neweigvals = 1. / (self.LRevals + regparam)
        self.A = self.svecs * np.multiply(1. / self.svals.T, (self.LRevecs * np.multiply(self.neweigvals.T, self.multipleright)))
        self.predictor = self.svdad.createModel(self)
        
    
    