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
from rlscore.learner.rls import RLS
from rlscore.utilities import array_tools
from rlscore.utilities import linalg


from . import _two_step_rls

from rlscore.predictor import PairwisePredictorInterface
from rlscore.predictor import LinearPairwisePredictor
from rlscore.predictor import KernelPairwisePredictor

class TwoStepRLS(PairwisePredictorInterface):

    """Two-step regularized least-squares regression with paired-input (dyadic) data.
    Closed form solution for complete data set with labels for all pairs known.
    

    Parameters
    ----------
    X1 : {array-like}, shape = [n_samples1, n_features1] 
        Data matrix 1 (for linear TwoStepRLS)
        
    X2 : {array-like}, shape = [n_samples2, n_features2] 
        Data matrix 2 (for linear TwoStepRLS)
        
    K1 : {array-like}, shape = [n_samples1, n_samples1]
        Kernel matrix 1 (for kernel TwoStepRLS)

    K2 : {array-like}, shape = [n_samples1, n_samples1]
        Kernel matrix 2 (for kernel TwoStepRLS)
        
    Y : {array-like}, shape = [n_samples1*n_samples2]
        Training set labels. Label for (X1[i], X2[j]) maps to
        Y[i + j*n_samples1] (column order).
        
    regparam1 : float
        regularization parameter 1, regparam1 > 0
        
    regparam2 : float
        regularization parameter 2, regparam2 > 0
        
    Attributes
    -----------
    predictor : {LinearPairwisePredictor, KernelPairwisePredictor}
        trained predictor
                  
    Notes
    -----
    
    Computational complexity of training:
    m = n_samples1, n = n_samples2, d = n_features1, e  = n_features2
    
    O(mnd + mne) Linear version (assumption: d < m, e < n)
    
    O(m^3 + n^3) Kernel version
     
    TwoStepRLS implements the closed form solution described in [1].
    
    
    References
    ----------
    
    [1] Tapio Pahikkala, Michiel Stock, Antti Airola, Tero Aittokallio, Bernard De Baets, and Willem Waegeman.
    A two-step learning approach for solving full and almost full cold start problems in dyadic prediction.
    Machine Learning and Knowledge Discovery in Databases (ECML PKDD 2014).
    Volume 8725 of Lecture Notes in Computer Science, pages 517--532. Springer, 2014.
    """
       
    
    def __init__(self, **kwargs):
        Y = kwargs["Y"]
        Y = np.mat(array_tools.as_2d_array(Y))
        if 'K1' in kwargs:
            K1 = np.mat(kwargs['K1'])
            K2 = np.mat(kwargs['K2'])
            Y = Y.reshape((K1.shape[0], K2.shape[0]), order = 'F')
            self.K1, self.K2 = K1, K2
            self.kernelmode = True
        else:
            X1 = np.mat(kwargs['X1'])
            X2 = np.mat(kwargs['X2'])
            Y = Y.reshape((X1.shape[0], X2.shape[0]), order = 'F')
            self.X1, self.X2 = X1, X2
            self.kernelmode = False
        self.Y = Y
        self.regparam1 = kwargs["regparam1"]
        self.regparam2 = kwargs["regparam2"]
        self.trained = False
        self.solve(self.regparam1, self.regparam2)
    
    
    def solve(self, regparam1, regparam2):
        """Re-trains TwoStepRLS for the given regparams
               
        Parameters
        ----------
        regparam1: float
            regularization parameter 1, regparam1 > 0
        
        regparam2: float
            regularization parameter 2, regparam2 > 0
            
        Notes
        -----    
                
        Computational complexity of re-training:
        
        m = n_samples1, n = n_samples2, d = n_features1, e  = n_features2
        
        O(ed^2 + de^2) Linear version (assumption: d < m, e < n)
        
        O(m^3 + n^3) Kernel version
        """
        self.regparam1 = regparam1
        self.regparam2 = regparam2
        if self.kernelmode:
            K1, K2 = self.K1, self.K2
            Y = self.Y.reshape((K1.shape[0], K2.shape[0]), order='F')
            if not self.trained:
                self.trained = True
                evals1, V  = linalg.eig_psd(K1)
                evals1 = np.mat(evals1).T
                evals1 = np.multiply(evals1, evals1)
                V = np.mat(V)
                self.evals1 = evals1
                self.V = V
                
                evals2, U = linalg.eig_psd(K2)
                evals2 = np.mat(evals2).T
                evals2 = np.multiply(evals2, evals2)
                U = np.mat(U)
                self.evals2 = evals2
                self.U = U
                self.VTYU = V.T * self.Y * U
            
            self.newevals1 = 1. / (self.evals1 + regparam1)
            self.newevals2 = 1. / (self.evals2 + regparam2)
            newevals = self.newevals1 * self.newevals2.T
            
            self.A = np.multiply(self.VTYU, newevals)
            self.A = self.V * self.A * self.U.T
            self.A = np.array(self.A)
            label_row_inds, label_col_inds = np.unravel_index(np.arange(K1.shape[0] * K2.shape[0]), (K1.shape[0],  K2.shape[0]))
            label_row_inds = np.array(label_row_inds, dtype = np.int32)
            label_col_inds = np.array(label_col_inds, dtype = np.int32)
            self.predictor = KernelPairwisePredictor(self.A.ravel(), label_row_inds, label_col_inds)
            
        else:
            X1, X2 = self.X1, self.X2
            Y = self.Y.reshape((X1.shape[0], X2.shape[0]), order='F')
            if not self.trained:
                self.trained = True
                V, svals1, rsvecs1 = linalg.svd_economy_sized(X1)
                svals1 = np.mat(svals1)
                self.svals1 = svals1.T
                self.evals1 = np.multiply(self.svals1, self.svals1)
                self.V = V
                self.rsvecs1 = np.mat(rsvecs1)
                
                if X1.shape == X2.shape and (X1 == X2).all():
                    svals2, U, rsvecs2 = svals1, V, rsvecs1
                else:
                    U, svals2, rsvecs2 = linalg.svd_economy_sized(X2)
                    svals2 = np.mat(svals2)
                self.svals2 = svals2.T
                self.evals2 = np.multiply(self.svals2, self.svals2)
                self.U = U
                self.rsvecs2 = np.mat(rsvecs2)
                
                self.VTYU = V.T * Y * U
            
            self.newevals1 = 1. / (self.evals1 + regparam1)
            self.newevals2 = 1. / (self.evals2 + regparam2)
            newevals = np.multiply(self.svals1, self.newevals1) * np.multiply(self.svals2, self.newevals2).T
            
            self.W = np.multiply(self.VTYU, newevals)
            self.W = self.rsvecs1.T * self.W * self.rsvecs2
            #self.predictor = LinearPairwisePredictor(self.W)
            self.predictor = LinearPairwisePredictor(np.array(self.W))
    
    
    def in_sample_loo(self):
        """
        Computes the in-sample leave-one-out cross-validation predictions. By in-sample we denote the
        setting, where we leave out one entry of Y at a time.
        
        Returns
        -------
        F : array, shape = [n_samples1*n_samples2]
            Training set labels. Label for (X1[i], X2[j]) maps to
            F[i + j*n_samples1] (column order).
            
        Notes
        -----    
                
        Computational complexity:
        
        m = n_samples1, n = n_samples2, d = n_features1, e  = n_features2
        
        O(mne + mnd) Linear version (assumption: d < m, e < n)
        
        O(mn^2 + m^2n) Kernel version
        """
        if not self.kernelmode:
            X1, X2 = self.X1, self.X2
            P = X1 * self.W * X2.T
        else:
            P = self.K1 * self.A * self.K2.T
        
        newevals = np.multiply(self.evals2 * self.evals1.T, 1. / ((self.evals2 + self.regparam2) * (self.evals1.T + self.regparam1)))
        Vsqr = np.multiply(self.V, self.V)
        Usqr = np.multiply(self.U, self.U)
        ccc = Vsqr * newevals.T * Usqr.T
        loopred = np.multiply(1. / (1. - ccc), P - np.multiply(ccc, self.Y))
        return np.asarray(loopred).ravel(order='F')
    
    
    def leave_x2_out(self):
        """
        Computes the leave-column-out cross-validation predictions. Here, all instances
        related to a single object from domain 2 are left out together at a time.
        
        Returns
        -------
        F : array, shape = [n_samples1*n_samples2]
            Training set labels. Label for (X1[i], X2[j]) maps to
            F[i + j*n_samples1] (column order).
        """
        
        VTY = self.V.T * self.Y
        
        filteredevals1 = self.evals1 / (self.evals1 + self.regparam1)
        
        foo = np.multiply(VTY, filteredevals1)
        foo = self.V * foo
        foo = np.array(foo)
        rlsparams = {}
        rlsparams["regparam"] = self.regparam2
        rlsparams["Y"] = foo.T
        rlsparams["bias"] = 0.
        if self.kernelmode:
            rlsparams["X"] = np.array(self.K2)
            rlsparams['kernel'] = 'PrecomputedKernel'
        else:
            rlsparams["X"] = np.array(self.X2)
        ordinary_rls_for_columns = RLS(**rlsparams)
        lco = ordinary_rls_for_columns.leave_one_out().T.ravel(order = 'F')
        return lco
    
    
    def leave_x1_out(self):
        """
        Computes the leave-row-out cross-validation predictions. Here, all instances
        related to a single object from domain 1 are left out together at a time.
        
        Returns
        -------
        F : array, shape = [n_samples1*n_samples2]
            Training set labels. Label for (X1[i], X2[j]) maps to
            F[i + j*n_samples1] (column order).
        """
        
        YU = self.Y * self.U
        
        filteredevals2 = self.evals2 / (self.evals2 + self.regparam2)
        
        foo = np.multiply(YU, filteredevals2.T)
        foo = foo * self.U.T
        foo = np.array(foo)
        rlsparams = {}
        rlsparams["regparam"] = self.regparam1
        rlsparams["Y"] = foo
        rlsparams["bias"] = 0.
        if self.kernelmode:
            rlsparams["X"] = np.array(self.K1)
            rlsparams['kernel'] = 'PrecomputedKernel'
        else:
            rlsparams["X"] = np.array(self.X1)
        ordinary_rls_for_rows = RLS(**rlsparams)
        lro = ordinary_rls_for_rows.leave_one_out().ravel(order = 'F')
        return lro
    
    
    def out_of_sample_loo(self):
        """
        Computes the out-of-sample cross-validation predictions. By out-of-sample we denote the
        setting, where when leaving out an entry (a,b) in Y, we also remove from training set
        all instances of type (a,x) and (x,b).
        
        Returns
        -------
        F : array, shape = [n_samples1*n_samples2]
            Training set labels. Label for (X1[i], X2[j]) maps to
            F[i + j*n_samples1] (column order).
            
        Notes
        -----    
                
        Computational complexity [TODO: check]:
        
        m = n_samples1, n = n_samples2, d = n_features1, e  = n_features2
        
        O(mne + mnd) Linear version (assumption: d < m, e < n)
        
        O(mn^2 + m^2n) Kernel version
        """
        
        bevals_col = np.multiply(self.evals2, self.newevals2).T
        
        svecsm_col = np.multiply(bevals_col, self.U)
        RQR_col = np.sum(np.multiply(self.U, svecsm_col), axis = 1)
        LOO_ek_col = (1. / (1. - RQR_col))
        LOO_col = (np.multiply(LOO_ek_col, self.U * (svecsm_col.T * self.Y.T)) - np.multiply(LOO_ek_col, np.multiply(RQR_col, self.Y.T))).T
        
        
        bevals_row = np.multiply(self.evals1, self.newevals1).T
        
        svecsm_row = np.multiply(bevals_row, self.V)
        RQR_row = np.sum(np.multiply(self.V, svecsm_row), axis = 1)
        LOO_ek_row = (1. / (1. - RQR_row))
        LOO_two_step = np.multiply(LOO_ek_row, self.V * (svecsm_row.T * LOO_col)) - np.multiply(LOO_ek_row, np.multiply(RQR_row, LOO_col))
        LOO_two_step = np.array(LOO_two_step)
        return LOO_two_step.ravel(order = 'F')
    
    
    def out_of_sample_loo_symmetric(self):
        """
        Computes the out-of-sample cross-validation predictions. By out-of-sample we denote the
        setting, where when leaving out an entry (a,b) in Y, we also remove from training set
        all instances of type (a,x) and (x,b). For symmetric data, where X1 = X2 (or K1 = K2).
        
        Returns
        -------
        F : array, shape = [n_samples1*n_samples2]
            Training set labels. Label for (X1[i], X2[j]) maps to
            F[i + j*n_samples1] (column order).
        """
        
        
        G = np.multiply((self.newevals1.T-(1./self.regparam1)), self.V) * self.V.T + (1./self.regparam1) * np.mat(np.identity(self.K1.shape[0]))
        GY = G * self.Y
        GYG = GY * G
        
        results = np.zeros((self.Y.shape[0], self.Y.shape[1]))
        _two_step_rls.out_of_sample_loo_symmetric(G, self.Y, GY, GYG, results, self.Y.shape[0], self.Y.shape[1])
        return results.ravel(order = 'F')


