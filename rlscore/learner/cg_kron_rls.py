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
from rlscore.utilities import pairwise_kernel_operator
from rlscore.predictor import PairwisePredictorInterface

CALLBACK_FUNCTION = 'callback'


class CGKronRLS(PairwisePredictorInterface):
    
    """Regularized least-squares regression with
    paired-input (dyadic) data and Kronecker kernels.
    Iterative solver for incomplete data set.
    

    Parameters
    ----------
    pko : {rlscore.utilities.PairwiseKernelOperator}, shape = [n_samples1, n_samples1], alternative to X1, X2, etc.
        Linear operator that determines the type of the Kronecker product kernel (for kernel CGKronRLS)..
    
    Y : {array-like}, shape = [n_train_pairs]
        Training set labels. 
        
    regparam : float, optional
        regularization parameter, regparam > 0 (default=1.0)
    
    X1 : {array-like}, shape = [n_samples1, n_features1]
        Data matrix 1 (for primal CGKronRLS)
        
    X2 : {array-like}, shape = [n_samples2, n_features2] 
        Data matrix 2 (for primal CGKronRLS)
        
    label_row_inds : {array-like, list of equal length array-likes}, shape = [n_train_pairs]
        row indices from X1, corresponding to labels in Y
    
    label_col_inds : {array-like, list of equal length array-likes}, shape = [n_train_pairs]
        row indices from X2, corresponding to labels in Y
    
    weights : {list, tuple, array-like}, shape = [n_kernels], optional
        weights used by multiple pairwise kernel predictors
    
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
        self.Y = kwargs["Y"]
        #self.Y = array_tools.as_2d_array(Y)
        self.trained = False
        if "regparam" in kwargs:
            self.regparam = kwargs["regparam"]
        else:
            self.regparam = 0.
        regparam = self.regparam
        if CALLBACK_FUNCTION in kwargs:
            self.callbackfun = kwargs[CALLBACK_FUNCTION]
        else:
            self.callbackfun = None
        if "compute_risk" in kwargs:
            self.compute_risk = kwargs["compute_risk"]
        else:
            self.compute_risk = False
        
        if 'K1' in kwargs or 'pko' in kwargs:
            if 'pko' in kwargs:
                pko = kwargs['pko']
            else:
                self.input1_inds = np.array(kwargs["label_row_inds"], dtype = np.int32)
                self.input2_inds = np.array(kwargs["label_col_inds"], dtype = np.int32)
                K1 = kwargs['K1']
                K2 = kwargs['K2']
                if 'weights' in kwargs: weights = kwargs['weights']
                else: weights = None
                pko = pairwise_kernel_operator.PairwiseKernelOperator(K1, K2, self.input1_inds, self.input2_inds, self.input1_inds, self.input2_inds, weights)
            self.pko = pko
            if 'maxiter' in kwargs: maxiter = int(kwargs['maxiter'])
            else: maxiter = None
            
            Y = np.array(self.Y).ravel(order = 'F')
            self.bestloss = float("inf")
            def mv(v):
                return pko.matvec(v) + regparam * v
            
            def mvr(v):
                raise Exception('This function should not be called!')
            
            def cgcb(v):
                if self.compute_risk:
                    P =  sampled_kronecker_products.sampled_vec_trick(v, K2, K1, self.input2_inds, self.input1_inds, self.input2_inds, self.input1_inds)
                    z = (Y - P)
                    Ka = sampled_kronecker_products.sampled_vec_trick(v, K2, K1, self.input2_inds, self.input1_inds, self.input2_inds, self.input1_inds)
                    loss = (np.dot(z, z) + regparam * np.dot(v, Ka))
                    print("loss", 0.5 * loss)
                    if loss < self.bestloss:
                        self.A = v.copy()
                        self.bestloss = loss
                else:
                    self.A = v
                if not self.callbackfun is None:
                    self.predictor = KernelPairwisePredictor(self.A, self.pko.col_inds_K1, self.pko.col_inds_K2, self.pko.weights)
                    self.callbackfun.callback(self)
            
            G = LinearOperator((self.Y.shape[0], self.Y.shape[0]), matvec = mv, rmatvec = mvr, dtype = np.float64)
            self.A = minres(G, self.Y, maxiter = maxiter, callback = cgcb, tol=1e-20)[0]
            self.predictor = KernelPairwisePredictor(self.A, self.pko.col_inds_K1, self.pko.col_inds_K2, self.pko.weights)
        else: #Primal case. Does not work with the operator interface yet.
            self.input1_inds = np.array(kwargs["label_row_inds"], dtype = np.int32)
            self.input2_inds = np.array(kwargs["label_col_inds"], dtype = np.int32)
            X1 = kwargs['X1']
            X2 = kwargs['X2']
            self.X1, self.X2 = X1, X2
            
            if 'maxiter' in kwargs: maxiter = int(kwargs['maxiter'])
            else: maxiter = None
            
            if isinstance(X1, (list, tuple)):
                raise NotImplementedError("Got list or tuple as X1 but multiple kernel learning has not been implemented for the primal case yet.")
                if 'weights' in kwargs: weights = kwargs['weights']
                else: weights = np.ones((len(X1)))
                x1tsize, x1fsize = X1[0].shape #m, d
                x2tsize, x2fsize = X2[0].shape #q, r
            else:
                weights = None
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
                raise Exception('This function should not be called!')
            
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
            if 'warm_start' in kwargs:
                x0 = np.array(kwargs['warm_start']).reshape(kronfcount, order = 'F')
            else:
                x0 = None
            minres(G, v_init, x0 = x0, maxiter = maxiter, callback = cgcb, tol=1e-20)[0].reshape((x1fsize, x2fsize), order='F')
            self.predictor = LinearPairwisePredictor(self.W, self.input1_inds, self.input2_inds, weights)
            if not self.callbackfun is None:
                    self.callbackfun.finished(self)

    



