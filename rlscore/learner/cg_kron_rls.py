#
# The MIT License (MIT)
#
# This file is part of RLScore 
#
# Copyright (c) 2013 - 2016 Tapio Pahikkala, Antti Airola
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

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import minres

from rlscore.predictor import LinearPairwisePredictor
from rlscore.predictor import KernelPairwisePredictor
from rlscore.utilities import array_tools
from rlscore.utilities import sampled_kronecker_products
from rlscore.predictor import PairwisePredictorInterface

CALLBACK_FUNCTION = 'callback'


class CGKronRLS(PairwisePredictorInterface):
    
    """Regularized least-squares regression with
    paired-input (dyadic) data and Kronecker kernels.
    Iterative solver for incomplete data set.
    

    Parameters
    ----------
    X1 : {array-like}, shape = [n_samples1, n_features1] 
        Data matrix 1 (for linear KronRLS)
        
    X2 : {array-like}, shape = [n_samples2, n_features2] 
        Data matrix 2 (for linear KronRLS)
        
    K1 : {array-like}, shape = [n_samples1, n_samples1]
        Kernel matrix 1 (for kernel KronRLS)

    K2 : {array-like}, shape = [n_samples1, n_samples1]
        Kernel matrix 2 (for kernel KronRLS)
        
    Y : {array-like}, shape = [n_train_pairs]
        Training set labels. 
        
    label_row_inds : list of indices, shape = [n_train_pairs]
        row indices from X1, corresponding to labels in Y
    
    label_col_inds : list of indices, shape = [n_train_pairs]
        row indices from X2, corresponding to labels in Y
        
    regparam : float, optional
        regularization parameter, regparam > 0 (default=1.0)
    
    maxiter : int, optional
        maximum number of iterations (default: no upper limit)
        
    Attributes
    -----------
    predictor : {LinearPairwisePredictor, KernelPairwisePredictor}
        trained predictor
                  
    Notes
    -----
    
    Computational complexity of training:

    TODO
     
    KronRLS implements the iterative algorithm described in [1], making use of an efficient
    sampled Kronecker product algorithm.
    
    
    References
    ----------
    
    [1] Tapio Pahikkala. 
    Fast gradient computation for learning with tensor product kernels and sparse training labels.
    Structural, Syntactic, and Statistical Pattern Recognition (S+SSPR).
    volume 8621 of Lecture Notes in Computer Science, pages 123--132. 2014.
    """
    
    
    def __init__(self, **kwargs):
        self.resource_pool = kwargs
        Y = kwargs["Y"]
        self.input1_inds = np.array(kwargs["label_row_inds"], dtype = np.int32)
        self.input2_inds = np.array(kwargs["label_col_inds"], dtype = np.int32)
        Y = array_tools.as_2d_array(Y)
        self.Y = np.mat(Y)
        self.trained = False
        if kwargs.has_key("regparam"):
            self.regparam = kwargs["regparam"]
        else:
            self.regparam = 0.
        if kwargs.has_key(CALLBACK_FUNCTION):
            self.callbackfun = kwargs[CALLBACK_FUNCTION]
        else:
            self.callbackfun = None
        if kwargs.has_key("compute_risk"):
            self.compute_risk = kwargs["compute_risk"]
        else:
            self.compute_risk = False
        
        regparam = self.regparam
        if self.resource_pool.has_key('K1'):
            
            K1 = self.resource_pool['K1']
            K2 = self.resource_pool['K2']
            
            if 'maxiter' in self.resource_pool: maxiter = int(self.resource_pool['maxiter'])
            else: maxiter = None
            
            Y = np.array(self.Y).ravel(order = 'F')
            self.bestloss = float("inf")
            def mv(v):
                return sampled_kronecker_products.sampled_vec_trick(v, K2, K1, self.input2_inds, self.input1_inds, self.input2_inds, self.input1_inds) + regparam * v
            
            def mvr(v):
                raise Exception('You should not be here!')
            
            def cgcb(v):
                if self.compute_risk:
                    P =  sampled_kronecker_products.sampled_vec_trick(v, K2, K1, self.input2_inds, self.input1_inds, self.input2_inds, self.input1_inds)
                    z = (Y - P)
                    Ka = sampled_kronecker_products.sampled_vec_trick(v, K2, K1, self.input2_inds, self.input1_inds, self.input2_inds, self.input1_inds)
                    loss = (np.dot(z,z)+regparam*np.dot(v,Ka))
                    print "loss", 0.5*loss
                    if loss < self.bestloss:
                        self.A = v.copy()
                        self.bestloss = loss
                else:
                    self.A = v
                if not self.callbackfun is None:
                    self.predictor = KernelPairwisePredictor(self.A, self.input1_inds, self.input2_inds)
                    self.callbackfun.callback(self)
    
            
            G = LinearOperator((len(self.input1_inds), len(self.input1_inds)), matvec = mv, rmatvec = mvr, dtype = np.float64)
            self. A = minres(G, self.Y, maxiter = maxiter, callback = cgcb, tol=1e-20)[0]
            self.predictor = KernelPairwisePredictor(self.A, self.input1_inds, self.input2_inds)
        else:
            X1 = self.resource_pool['X1']
            X2 = self.resource_pool['X2']
            self.X1, self.X2 = X1, X2
            
            if 'maxiter' in self.resource_pool: maxiter = int(self.resource_pool['maxiter'])
            else: maxiter = None
            
            x1tsize, x1fsize = X1.shape #m, d
            x2tsize, x2fsize = X2.shape #q, r
            
            kronfcount = x1fsize * x2fsize
            
            Y = np.array(self.Y).ravel(order = 'F')
            self.bestloss = float("inf")
            def mv(v):
                v_after = sampled_kronecker_products.sampled_vec_trick(v, X2, X1, self.input2_inds, self.input1_inds)
                v_after = sampled_kronecker_products.sampled_vec_trick(v_after, X2.T, X1.T, None, None, self.input2_inds, self.input1_inds) + regparam * v
                return v_after
            
            def mvr(v):
                raise Exception('You should not be here!')
                return None
            
            def cgcb(v):
                if self.compute_risk:
                    P = sampled_kronecker_products.sampled_vec_trick(v, X2, X1, self.input2_inds, self.input1_inds)
                    z = (Y - P)
                    loss = (np.dot(z,z)+regparam*np.dot(v,v))
                    if loss < self.bestloss:
                        self.W = v.copy().reshape((x1fsize, x2fsize), order = 'F')
                        self.bestloss = loss
                else:
                    self.W = v.reshape((x1fsize, x2fsize), order = 'F')
                if not self.callbackfun is None:
                    self.predictor = LinearPairwisePredictor(self.W)
                    self.callbackfun.callback(self)
                
            G = LinearOperator((kronfcount, kronfcount), matvec = mv, rmatvec = mvr, dtype = np.float64)
            
            v_init = np.array(self.Y).reshape(self.Y.shape[0])
            v_init = sampled_kronecker_products.sampled_vec_trick(v_init, X2.T, X1.T, None, None, self.input2_inds, self.input1_inds)
            v_init = np.array(v_init).reshape(kronfcount)
            if self.resource_pool.has_key('warm_start'):
                x0 = np.array(self.resource_pool['warm_start']).reshape(kronfcount, order = 'F')
            else:
                x0 = None
            minres(G, v_init, x0 = x0, maxiter = maxiter, callback = cgcb, tol=1e-20)[0].reshape((x1fsize, x2fsize), order='F')
            self.predictor = LinearPairwisePredictor(self.W)
            if not self.callbackfun is None:
                    self.callbackfun.finished(self)

    



