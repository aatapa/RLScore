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
        
    K1 : {array-like, list of equally shaped array-likes}, shape = [n_samples1, n_samples1]
        Kernel matrix 1 (for kernel KronRLS)

    K2 : {array-like, list of equally shaped array-likes}, shape = [n_samples1, n_samples1]
        Kernel matrix 2 (for kernel KronRLS)
        
    weights : {list, tuple, array-like}, shape = [n_kernels], optional
        weights used by multiple pairwise kernel predictors
    
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
        Y = kwargs["Y"]
        self.input1_inds = np.array(kwargs["label_row_inds"], dtype = np.int32)
        self.input2_inds = np.array(kwargs["label_col_inds"], dtype = np.int32)
        Y = array_tools.as_2d_array(Y)
        self.Y = np.mat(Y)
        self.trained = False
        if "regparam" in kwargs:
            self.regparam = kwargs["regparam"]
        else:
            self.regparam = 0.
        if CALLBACK_FUNCTION in kwargs:
            self.callbackfun = kwargs[CALLBACK_FUNCTION]
        else:
            self.callbackfun = None
        if "compute_risk" in kwargs:
            self.compute_risk = kwargs["compute_risk"]
        else:
            self.compute_risk = False
        
        regparam = self.regparam
        if 'K1' in kwargs:
            
            K1 = kwargs['K1']
            K2 = kwargs['K2']
            
            if 'maxiter' in kwargs: maxiter = int(kwargs['maxiter'])
            else: maxiter = None
            
            Y = np.array(self.Y).ravel(order = 'F')
            self.bestloss = float("inf")
            def mv(v):
                return sampled_kronecker_products.sampled_vec_trick(v, K2, K1, self.input2_inds, self.input1_inds, self.input2_inds, self.input1_inds) + regparam * v
            def mv_mk(v):
                vsum = regparam * v
                for i in range(len(K1)):
                    K1i = K1[i]
                    K2i = K2[i]
                    vsum += weights[i] * sampled_kronecker_products.sampled_vec_trick(v, K2i, K1i, self.input2_inds, self.input1_inds, self.input2_inds, self.input1_inds)
                return vsum
            
            def mvr(v):
                raise Exception('You should not be here!')
            
            def cgcb(v):
                if self.compute_risk:
                    P =  sampled_kronecker_products.sampled_vec_trick(v, K2, K1, self.input2_inds, self.input1_inds, self.input2_inds, self.input1_inds)
                    z = (Y - P)
                    Ka = sampled_kronecker_products.sampled_vec_trick(v, K2, K1, self.input2_inds, self.input1_inds, self.input2_inds, self.input1_inds)
                    loss = (np.dot(z,z)+regparam*np.dot(v,Ka))
                    print("loss", 0.5*loss)
                    if loss < self.bestloss:
                        self.A = v.copy()
                        self.bestloss = loss
                else:
                    self.A = v
                if not self.callbackfun is None:
                    self.predictor = KernelPairwisePredictor(self.A, self.input1_inds, self.input2_inds)
                    self.callbackfun.callback(self)
            
            if isinstance(K1, (list, tuple)):
                if 'weights' in kwargs: weights = kwargs['weights']
                else: weights = np.ones((len(K1)))
                G = LinearOperator((len(self.input1_inds), len(self.input1_inds)), matvec = mv_mk, rmatvec = mvr, dtype = np.float64)
            else:
                weights = None
                G = LinearOperator((len(self.input1_inds), len(self.input1_inds)), matvec = mv, rmatvec = mvr, dtype = np.float64)
            self. A = minres(G, self.Y, maxiter = maxiter, callback = cgcb, tol=1e-20)[0]
            self.predictor = KernelPairwisePredictor(self.A, self.input1_inds, self.input2_inds, weights)
        else:
            X1 = kwargs['X1']
            X2 = kwargs['X2']
            self.X1, self.X2 = X1, X2
            
            if 'maxiter' in kwargs: maxiter = int(kwargs['maxiter'])
            else: maxiter = None
            
            if isinstance(X1, (list, tuple)):
                raise NotImplementedError("Got list or tuple as X1 but multiple kernel learning has not been implemented for the proal case yet.")
                x1tsize, x1fsize = X1[0].shape #m, d
                x2tsize, x2fsize = X2[0].shape #q, r
            else:
                x1tsize, x1fsize = X1.shape #m, d
                x2tsize, x2fsize = X2.shape #q, r
            
            kronfcount = x1fsize * x2fsize
            
            Y = np.array(self.Y).ravel(order = 'F')
            self.bestloss = float("inf")
            def mv(v):
                v_after = sampled_kronecker_products.sampled_vec_trick(v, X2, X1, self.input2_inds, self.input1_inds)
                v_after = sampled_kronecker_products.sampled_vec_trick(v_after, X2.T, X1.T, None, None, self.input2_inds, self.input1_inds) + regparam * v
                return v_after
            def mv_mk(v):
                vsum = regparam * v
                for i in range(len(X1)):
                    X1i = X1[i]
                    X2i = X2[i]
                    v_after = sampled_kronecker_products.sampled_vec_trick(v, X2i, X1i, self.input2_inds, self.input1_inds)
                    v_after = sampled_kronecker_products.sampled_vec_trick(v_after, X2i.T, X1i.T, None, None, self.input2_inds, self.input1_inds)
                    vsum = vsum + v_after
                return vsum
            
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
            
            if isinstance(X1, (list, tuple)):
                G = LinearOperator((kronfcount, kronfcount), matvec = mv_mk, rmatvec = mvr, dtype = np.float64)
                vsum = np.zeros(kronfcount)
                v_init = np.array(self.Y).reshape(self.Y.shape[0])
                for i in range(len(X1)):
                    X1i = X1[i]
                    X2i = X2[i]
                    vsum += sampled_kronecker_products.sampled_vec_trick(v_init, X2i.T, X1i.T, None, None, self.input2_inds, self.input1_inds)
                v_init = vsum
            else:
                G = LinearOperator((kronfcount, kronfcount), matvec = mv, rmatvec = mvr, dtype = np.float64)
                v_init = np.array(self.Y).reshape(self.Y.shape[0])
                v_init = sampled_kronecker_products.sampled_vec_trick(v_init, X2.T, X1.T, None, None, self.input2_inds, self.input1_inds)
            
            v_init = np.array(v_init).reshape(kronfcount)
            if 'warm_start' in kwargs:
                x0 = np.array(kwargs['warm_start']).reshape(kronfcount, order = 'F')
            else:
                x0 = None
            minres(G, v_init, x0 = x0, maxiter = maxiter, callback = cgcb, tol=1e-20)[0].reshape((x1fsize, x2fsize), order='F')
            self.predictor = LinearPairwisePredictor(self.W)
            if not self.callbackfun is None:
                    self.callbackfun.finished(self)

    



