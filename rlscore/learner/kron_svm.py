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

import numpy as np

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import qmr

from rlscore.utilities import sampled_kronecker_products
from rlscore.predictor import KernelPairwisePredictor
from rlscore.predictor import LinearPairwisePredictor
from rlscore.predictor import PairwisePredictorInterface


TRAIN_LABELS = 'Y'
CALLBACK_FUNCTION = 'callback'
        

def func(v, X1, X2, Y, rowind, colind, lamb):
    P = sampled_kronecker_products.sampled_vec_trick(v, X2, X1, colind, rowind)
    z = (1. - Y*P)
    z = np.where(z>0, z, 0)
    #return np.dot(z,z)
    return 0.5*(np.dot(z,z)+lamb*np.dot(v,v))

def gradient(v, X1, X2, Y, rowind, colind, lamb):
    P = sampled_kronecker_products.sampled_vec_trick(v, X2, X1, colind, rowind)
    z = (1. - Y*P)
    z = np.where(z>0, z, 0)
    sv = np.nonzero(z)[0]
    rows = rowind[sv]
    cols = colind[sv]
    A = - sampled_kronecker_products.sampled_vec_trick(Y[sv], X2.T, X1.T, None, None, cols, rows)
    B = sampled_kronecker_products.sampled_vec_trick(P[sv], X2.T, X1.T, None, None, cols, rows)
    return A + B + lamb*v

def hessian(v, p, X1, X2, Y, rowind, colind, lamb):
    P = sampled_kronecker_products.sampled_vec_trick(v, X2, X1, colind, rowind)
    z = (1. - Y*P)
    z = np.where(z>0, z, 0)
    sv = np.nonzero(z)[0]
    rows = rowind[sv]
    cols = colind[sv]
    p_after = sampled_kronecker_products.sampled_vec_trick(p, X2, X1, cols, rows)
    p_after = sampled_kronecker_products.sampled_vec_trick(p_after, X2.T, X1.T, None, None, cols, rows)
    return p_after + lamb*p    

class KronSVM(PairwisePredictorInterface):
        
    
    def __init__(self, **kwargs):
        self.resource_pool = kwargs
        Y = kwargs[TRAIN_LABELS]
        self.label_row_inds = np.array(kwargs["label_row_inds"], dtype = np.int32)
        self.label_col_inds = np.array(kwargs["label_col_inds"], dtype = np.int32)
        self.Y = Y
        self.trained = False
        if "regparam" in kwargs:
            self.regparam = kwargs["regparam"]
        else:
            self.regparam = 1.0
        if CALLBACK_FUNCTION in kwargs:
            self.callbackfun = kwargs[CALLBACK_FUNCTION]
        else:
            self.callbackfun = None
        if "compute_risk" in kwargs:
            self.compute_risk = kwargs["compute_risk"]
        else:
            self.compute_risk = False
        
        regparam = self.regparam
        
        if not 'K1' in self.resource_pool:
            self.regparam = regparam
            X1 = self.resource_pool['X1']
            X2 = self.resource_pool['X2']
            self.X1, self.X2 = X1, X2
            
            if 'maxiter' in self.resource_pool: maxiter = int(self.resource_pool['maxiter'])
            else: maxiter = 1000
    
            if 'inneriter' in self.resource_pool: inneriter = int(self.resource_pool['inneriter'])
            else: inneriter = 50
            
            x1tsize, x1fsize = X1.shape #m, d
            x2tsize, x2fsize = X2.shape #q, r
            
            label_row_inds = np.array(self.label_row_inds, dtype = np.int32)
            label_col_inds = np.array(self.label_col_inds, dtype = np.int32)
            
    
    
            Y = self.Y
            rowind = label_row_inds
            colind = label_col_inds
            lamb = self.regparam
            rowind = np.array(rowind, dtype = np.int32)
            colind = np.array(colind, dtype = np.int32)
            fdim = X1.shape[1]*X2.shape[1]
            w = np.zeros(fdim)
            #np.random.seed(1)
            #w = np.random.random(fdim)
            self.bestloss = float("inf")
            def mv(v):
                return hessian(w, v, X1, X2, Y, rowind, colind, lamb)
                
            for i in range(maxiter):
                g = gradient(w, X1, X2, Y, rowind, colind, lamb)
                G = LinearOperator((fdim, fdim), matvec=mv, rmatvec=mv, dtype=np.float64)
                self.best_residual = float("inf")
                self.w_new = qmr(G, g, tol=1e-10, maxiter=inneriter)[0]
                if np.all(w == w - self.w_new):
                    break
                w = w - self.w_new
                if self.compute_risk:
                    P = sampled_kronecker_products.sampled_vec_trick(w, X1, X2, rowind, colind)
                    z = (1. - Y*P)
                    z = np.where(z>0, z, 0)
                    loss = 0.5*(np.dot(z,z)+lamb*np.dot(w,w))
                    if loss < self.bestloss:
                        self.W = w.reshape((x1fsize, x2fsize), order='F')
                        self.bestloss = loss
                else:
                    self.W = w.reshape((x1fsize, x2fsize), order='F')             
                if self.callbackfun is not None:
                    self.callbackfun.callback(self)
            self.predictor = LinearPairwisePredictor(self.W)
        else:
            K1 = self.resource_pool['K1']
            K2 = self.resource_pool['K2']
            if 'maxiter' in self.resource_pool: maxiter = int(self.resource_pool['maxiter'])
            else: maxiter = 100
            if 'inneriter' in self.resource_pool: inneriter = int(self.resource_pool['inneriter'])
            else: inneriter = 1000
            label_row_inds = np.array(self.label_row_inds, dtype = np.int32)
            label_col_inds = np.array(self.label_col_inds, dtype = np.int32)
            
            Y = self.Y
            rowind = label_row_inds
            colind = label_col_inds
            lamb = self.regparam
            rowind = np.array(rowind, dtype = np.int32)
            colind = np.array(colind, dtype = np.int32)
            ddim = len(rowind)
            a = np.zeros(ddim)
            self.bestloss = float("inf")
            
            def func(a):
                P =  sampled_kronecker_products.sampled_vec_trick(a, K2, K1, colind, rowind, colind, rowind)
                z = (1. - Y*P)
                z = np.where(z>0, z, 0)
                Ka = sampled_kronecker_products.sampled_vec_trick(a, K2, K1, colind, rowind, colind, rowind)
                return 0.5*(np.dot(z,z)+lamb*np.dot(a, Ka))
            
            def mv(v):
                rows = rowind[sv]
                cols = colind[sv]
                p = np.zeros(len(rowind))
                A = sampled_kronecker_products.sampled_vec_trick(v, K2, K1, cols, rows, colind, rowind)
                p[sv] = A
                return p + lamb * v
            def mv_mk(v):
                rows = rowind[sv]
                cols = colind[sv]
                p = np.zeros(len(rowind))
                skpsum = np.zeros(len(sv))
                for i in range(len(K1)):
                    K1i = K1[i]
                    K2i = K2[i]
                    skpsum += weights[i] * sampled_kronecker_products.sampled_vec_trick(v, K2i, K1i, cols, rows, colind, rowind)
                p[sv] = skpsum
                return p + lamb * v
            
            def rv(v):
                rows = rowind[sv]
                cols = colind[sv]
                p = sampled_kronecker_products.sampled_vec_trick(v[sv], K2, K1, colind, rowind, cols, rows)
                return p + lamb * v
            
            def rv_mk(v):
                rows = rowind[sv]
                cols = colind[sv]
                psum = np.zeros(len(v))
                for i in range(len(K1)):
                    K1i = K1[i]
                    K2i = K2[i]
                    psum += weights[i] * sampled_kronecker_products.sampled_vec_trick(v[sv], K2i, K1i, colind, rowind, cols, rows)
                return psum + lamb * v
            
            for i in range(maxiter):
                if isinstance(K1, (list, tuple)):
                    if 'weights' in kwargs: weights = kwargs['weights']
                    else: weights = np.ones((len(K1)))
                    A = LinearOperator((ddim, ddim), matvec = mv_mk, rmatvec = rv_mk, dtype = np.float64)
                    P = np.zeros(ddim)
                    for i in range(len(K1)):
                        K1i = K1[i]
                        K2i = K2[i]
                        prod_i = weights[i] * sampled_kronecker_products.sampled_vec_trick(a, K2i, K1i, colind, rowind, colind, rowind)
                        P += prod_i
                else:
                    weights = None
                    A = LinearOperator((ddim, ddim), matvec=mv, rmatvec=rv, dtype=np.float64)
                    P = sampled_kronecker_products.sampled_vec_trick(a, K2, K1, colind, rowind, colind, rowind)
                z = (1. - Y*P)
                z = np.where(z>0, z, 0)
                sv = np.nonzero(z)[0]
                B = np.zeros(P.shape)
                B[sv] = P[sv]-Y[sv]
                B = B + lamb*a
                #solve Ax = B
                self.a_new = qmr(A, B, tol=1e-10, maxiter=inneriter)[0]
                if np.all(a == a - self.a_new):
                    break
                a = a - self.a_new
                if self.compute_risk:
                    loss = func(a)
                    if loss < self.bestloss:
                        self.A = a
                        self.bestloss = loss
                else:
                    self.A = a
                self.predictor = KernelPairwisePredictor(a, rowind, colind)
                if self.callbackfun is not None:
                    self.callbackfun.callback(self)
            self.predictor = KernelPairwisePredictor(a, rowind, colind)
            if self.callbackfun is not None:
                self.callbackfun.finished(self)
