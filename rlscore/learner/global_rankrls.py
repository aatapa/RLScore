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
from numpy import array, eye, float64, multiply, mat, ones, sqrt, zeros
import numpy.linalg as la

from rlscore.utilities import array_tools
from rlscore.utilities import adapter
from rlscore.predictor import PredictorInterface
from rlscore.utilities.cross_validation import grid_search
from rlscore.measure import cindex
from rlscore.learner.rls import NfoldCV
from rlscore.learner.rls import LPOCV

from . import _global_rankrls

class GlobalRankRLS(PredictorInterface):
    """RankRLS: Regularized least-squares ranking.
    Global ranking (see QueryRankRLS for query-structured data)

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Data matrix
    Y : {array-like}, shape = [n_samples] or [n_samples, n_labels]
        Training set labels
    regparam : float, optional
        regularization parameter, regparam > 0 (default=1.0)
    kernel : {'LinearKernel', 'GaussianKernel', 'PolynomialKernel', 'PrecomputedKernel', ...}
        kernel function name, imported dynamically from rlscore.kernel
    basis_vectors : {array-like, sparse matrix}, shape = [n_bvectors, n_features], optional
        basis vectors (typically a randomly chosen subset of the training data)
        
    Other Parameters
    ----------------
    Typical kernel parameters include:
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
    m = n_samples, d = n_features, l = n_labels, b = n_bvectors
    
    O(m^3 + dm^2 + lm^2): basic case
    
    O(md^2 +lmd): Linear Kernel, d < m
    
    O(mb^2 +lmb): Sparse approximation with basis vectors 
    
     
    RankRLS algorithm is described in [1,2]. The leave-pair-out cross-validation algorithm
    is described in [2,3]. The use of leave-pair-out cross-validation for AUC estimation
    is analyzed in [4].
    
    References
    ----------
    
    [1] Tapio Pahikkala, Evgeni Tsivtsivadze, Antti Airola, Jorma Boberg and Tapio Salakoski
    Learning to rank with pairwise regularized least-squares.
    In Thorsten Joachims, Hang Li, Tie-Yan Liu, and ChengXiang Zhai, editors,
    SIGIR 2007 Workshop on Learning to Rank for Information Retrieval, pages 27--33, 2007.
    
    [2] Tapio Pahikkala, Evgeni Tsivtsivadze, Antti Airola, Jouni Jarvinen, and Jorma Boberg.
    An efficient algorithm for learning to rank from preference graphs.
    Machine Learning, 75(1):129-165, 2009.
    
    [3] Tapio Pahikkala, Antti Airola, Jorma Boberg, and Tapio Salakoski.
    Exact and efficient leave-pair-out cross-validation for ranking RLS.
    In Proceedings of the 2nd International and Interdisciplinary Conference
    on Adaptive Knowledge Representation and Reasoning (AKRR'08), pages 1-8,
    Espoo, Finland, 2008.

    [4] Antti Airola, Tapio Pahikkala, Willem Waegeman, Bernard De Baets, Tapio Salakoski.
    An Experimental Comparison of Cross-Validation Techniques for Estimating the Area Under the ROC Curve.
    Computational Statistics & Data Analysis 55(4), 1828-1844, 2011.

    """
    
    def __init__(self, X, Y, regparam = 1.0, kernel='LinearKernel', basis_vectors = None, **kwargs):
        Y = array_tools.as_2d_array(Y)
        self.Y = np.mat(Y)
        if X.shape[0] != Y.shape[0]:
            raise Exception("First dimension of X and Y must be the same")
        if basis_vectors is not None:
            if X.shape[1] != basis_vectors.shape[1]:
                raise Exception("Number of columns for X and basis_vectors must be the same")
        kwargs["bias"] = 0.
        kwargs['kernel'] =  kernel
        kwargs['X'] = X
        if basis_vectors is not None:
            kwargs['basis_vectors'] = basis_vectors
        self.svdad = adapter.createSVDAdapter(**kwargs)
        self.regparam = regparam
        self.svals = np.mat(self.svdad.svals)
        self.svecs = np.mat(self.svdad.rsvecs)
        self.size = self.Y.shape[0]
        self.solve(self.regparam)
    
    def solve(self, regparam=1.0):
        """Re-trains RankRLS for the given regparam
               
        Parameters
        ----------
        regparam : float, optional
            regularization parameter, regparam > 0 (default=1.0)
            
        Notes
        -----
    
        Computational complexity of re-training:
        m = n_samples, d = n_features, l = n_labels, b = n_bvectors
        
        O(lm^2): basic case
        
        O(lmd): Linear Kernel, d < m
        
        O(lmb): Sparse approximation with basis vectors 
             
        """
        if not hasattr(self, "multiplyright"):
            
            #Eigenvalues of the kernel matrix
            self.evals = multiply(self.svals, self.svals)
            
            #Temporary variables
            ssvecs = multiply(self.svecs, self.svals)
            J = mat(ones((self.size, 1), dtype=float64))
            
            #These are cached for later use in solve function
            self.lowrankchange = ssvecs.T * J[range(ssvecs.shape[0])]
            self.multipleright = ssvecs.T * (self.size * self.Y - J * (J.T * self.Y))
        
        self.regparam = regparam
        
        #Compute the eigenvalues determined by the given regularization parameter
        neweigvals = 1. / (self.size * self.evals + regparam)
        
        P = self.lowrankchange
        nP = multiply(neweigvals.T, P)
        fastinv = 1. / (-1. + P.T * nP)
        self.A = self.svecs * multiply(1. / self.svals.T, \
            (multiply(neweigvals.T, self.multipleright) \
            - nP * (fastinv * (nP.T * self.multipleright))))
        self.predictor = self.svdad.createModel(self)
    
    
    def leave_pair_out(self, pairs_start_inds, pairs_end_inds):
        """Computes leave-pair-out predictions for a trained RankRLS.
        
        Parameters
        ----------
        pairs_start_inds : list of indices, shape = [n_pairs]
            list of indices from range [0, n_samples-1]
        pairs_end_inds : list of indices, shape = [n_pairs]
            list of indices from range [0, n_samples-1]
        
        Returns
        -------
        P1 : array, shape = [n_pairs]
            holdout predictions for pairs_start_inds
        P2 : array, shape = [n_pairs]
            holdout predictions for pairs_end_inds
            
        Notes
        -----
    
        Computes the leave-pair-out cross-validation predictions, where each (i,j) pair with
        i= pair_start_inds[k] and j = pairs_end_inds[k] is left out in turn.
        
        When estimating area under ROC curve with leave-pair-out, one should leave out all
        positive-negative pairs, while for estimating the general ranking error one should
        leave out all pairs with different labels.
        
        Computational complexity of leave-pair-out with most pairs left out:
        m = n_samples, d = n_features, l = n_labels, b = n_bvectors
        
        O(lm^2+m^3): basic case
        
        O(lm^2+dm^2): Linear Kernel, d < m
        
        O(lm^2+bm^2): Sparse approximation with basis vectors 
        
        The leave-pair-out cross-validation algorithm is described in [1,2]. The use of
        leave-pair-out cross-validation for AUC estimation has been analyzed in [3]
        
        [1] Tapio Pahikkala, Evgeni Tsivtsivadze, Antti Airola, Jouni Jarvinen, and Jorma Boberg.
        An efficient algorithm for learning to rank from preference graphs.
        Machine Learning, 75(1):129-165, 2009.
    
        [2] Tapio Pahikkala, Antti Airola, Jorma Boberg, and Tapio Salakoski.
        Exact and efficient leave-pair-out cross-validation for ranking RLS.
        In Proceedings of the 2nd International and Interdisciplinary Conference
        on Adaptive Knowledge Representation and Reasoning (AKRR'08), pages 1-8,
        Espoo, Finland, 2008.

        [3] Antti Airola, Tapio Pahikkala, Willem Waegeman, Bernard De Baets, Tapio Salakoski.
        An Experimental Comparison of Cross-Validation Techniques for Estimating the Area Under the ROC Curve.
        Computational Statistics & Data Analysis 55(4), 1828-1844, 2011.
        """
        pairs_start_inds = array_tools.as_index_list(pairs_start_inds, self.Y.shape[0])
        pairs_end_inds = array_tools.as_index_list(pairs_end_inds, self.Y.shape[0])        
        
        evals, svecs = self.evals, self.svecs
        m = self.size
        
        Y = self.Y
        
        modevals = np.squeeze(np.array(np.multiply(evals, 1. / ((m - 2.) * evals + self.regparam))))
        GDY = (self.size - 2.) * (svecs * np.multiply(np.mat(modevals).T, (svecs.T * Y)))
        GC = np.squeeze(np.array(svecs * np.multiply(np.mat(modevals).T, np.sum(svecs.T, axis = 1))))
        CTGC = np.sum(GC)
        
        pairslen = len(pairs_start_inds)
        sm2Gdiag = np.zeros((self.Y.shape[0]))
        BTY = np.zeros((self.Y.shape))
        sqrtsm2GDY = np.zeros((self.Y.shape))
        BTGBBTY = np.zeros((self.Y.shape))
        results_first = np.zeros((pairslen, self.Y.shape[1]))
        results_second = np.zeros((pairslen, self.Y.shape[1]))
        
        _global_rankrls.leave_pair_out(pairslen,
                                                             self.Y.shape[0],
                                                             pairs_start_inds,
                                                             pairs_end_inds,
                                                             self.Y.shape[1],
                                                             Y,
                                                             svecs,
                                                             modevals,
                                                             svecs.shape[1],
                                                             np.zeros((self.Y.shape[0])),
                                                             np.squeeze(np.array(GC)),
                                                             sm2Gdiag,
                                                             CTGC,
                                                             GDY,
                                                             BTY,
                                                             sqrtsm2GDY,
                                                             BTGBBTY,
                                                             np.array(np.sum(Y, axis = 0))[0], #CTY
                                                             np.array(np.sum(GDY, axis = 0))[0], #CTGDY
                                                             results_first,
                                                             results_second)        
        
        return np.squeeze(results_first), np.squeeze(results_second)
    
    
    def _leave_pair_out_python(self, pairs_start_inds, pairs_end_inds, oind=0):
        """Computes leave-pair-out predictions for a trained RankRLS.
        
        Parameters
        ----------
        pairs_start_inds : list of indices, shape = [n_pairs]
            list of indices from range [0, n_samples-1]
        pairs_end_inds : list of indices, shape = [n_pairs]
            list of indices from range [0, n_samples-1]
        oind : index from range [0, n_samples-1]
            column of Y, for which pairwise cv is computed
        
        Returns
        -------
        P1 : array, shape = [n_pairs]
            holdout predictions for pairs_start_inds
        P2 : array, shape = [n_pairs]
            holdout predictions for pairs_end_inds
            
        Notes
        -----
    
        Computes the leave-pair-out cross-validation predicitons, where each (i,j) pair with
        i= pair_start_inds[k] and j = pairs_end_inds[k] is left out in turn.
        
        When estimating area under ROC curve with leave-pair-out, one should leave out all
        positive-negative pairs, while for estimating the general ranking error one should
        leave out all pairs with different labels.
        
        Computational complexity of holdout with most pairs left out: m = n_samples
        
        O(TODO)
        
        The leave-pair-out cross-validation algorithm is described in [1,2]. The use of
        leave-pair-out cross-validation for AUC estimation has been analyzed in [3]

        References
        ----------
        
        [1] Tapio Pahikkala, Evgeni Tsivtsivadze, Antti Airola, Jouni Jarvinen, and Jorma Boberg.
        An efficient algorithm for learning to rank from preference graphs.
        Machine Learning, 75(1):129-165, 2009.
    
        [2] Tapio Pahikkala, Antti Airola, Jorma Boberg, and Tapio Salakoski.
        Exact and efficient leave-pair-out cross-validation for ranking RLS.
        In Proceedings of the 2nd International and Interdisciplinary Conference
        on Adaptive Knowledge Representation and Reasoning (AKRR'08), pages 1-8,
        Espoo, Finland, 2008.

        [3] Antti Airola, Tapio Pahikkala, Willem Waegeman, Bernard De Baets, Tapio Salakoski.
        An Experimental Comparison of Cross-Validation Techniques for Estimating the Area Under the ROC Curve.
        Computational Statistics & Data Analysis 55(4), 1828-1844, 2011.
        """
        
        evals, svecs = self.evals, self.svecs
        m = self.size
        
        Y = self.Y
        
        #This is, in the worst case, a cubic operation.
        #If there are multiple outputs,
        #this operation should be common for them all. THIS IS TO BE FIXED!
        def computeG():
            regparam = self.regparam
            G = svecs * multiply(multiply(evals, 1. / ((m - 2.) * evals + regparam)).T, svecs.T)
            return G
        G = computeG()
        
        GDY = (self.size - 2.) * G * Y
        GC = sum(G, axis=1)
        
        CTGC = sum(GC)

        
        CTY = sum(Y, axis=0)[0, oind]
        CTGDY = sum(GDY, axis=0)[0, oind]
        
        sm2 = self.size - 2.
        sqrtsm2 = sqrt(sm2)
        
        #Array is faster to access than matrix
        G = array(G)
        
        #Lists are faster to access than matrices or arrays
        def hack():
            GDY_ = []
            sqrtsm2GDY_ = []
            GC_ = []
            Y_ = []
            BTY_ = []
            Gdiag_ = []
            sm2Gdiag_ = []
            BTGBBTY_ = []
            for i in range(m):
                GDYi = GDY[i, oind]
                GDY_.append(GDYi)
                sqrtsm2GDY_.append(sqrtsm2 * GDYi)
                GC_.append(GC[i, 0])
                Yi = Y[i, oind]
                Y_.append(Yi)
                BTY_.append(sqrtsm2 * Yi)
                Gii = G[i, i]
                Gdiag_.append(Gii)
                sm2Gdiag_.append(sm2 * Gii - 1.)
                BTGBBTY_.append(sm2 * Gii * sqrtsm2 * Yi)
            return GDY_, sqrtsm2GDY_, GC_, Y_, BTY_, Gdiag_, sm2Gdiag_, BTGBBTY_
        GDY_, sqrtsm2GDY_, GC_, Y_, BTY_, Gdiag_, sm2Gdiag_, BTGBBTY_ = hack()
        
        results_start, results_end = [], []
        
        #This loops through the list of hold-out pairs.
        #Each pair is handled in a constant time.
        def looppairs(results_start, results_end):
            for pairind in range(len(pairs_start_inds)):
                
                i, j = pairs_start_inds[pairind], pairs_end_inds[pairind]
                
                Gii = Gdiag_[i]
                Gij = G[i, j]
                Gjj = Gdiag_[j]
                GCi = GC_[i]
                GCj = GC_[j]
                
                Yi = Y_[i]
                Yj = Y_[j]
                
                GDYi = GDY_[i]
                GDYj = GDY_[j]
                
                BTY0 = CTY - Yi - Yj
                BTY1 = BTY_[i]
                BTY2 = BTY_[j]
                
                GiipGij = Gii + Gij
                GijpGjj = Gij + Gjj
                GCipGCj = GCi + GCj
                
                BTGB00 = GiipGij + GijpGjj + CTGC - GCipGCj - GCipGCj
                BTGB01 = sqrtsm2 * (GCi - GiipGij)
                BTGB02 = sqrtsm2 * (GCj - GijpGjj)
                BTGB12 = sm2 * Gij
                
                BTGLY0 = CTGDY - (GDYi + GDYj + BTGB00 * BTY0 + BTGB01 * BTY1 + BTGB02 * BTY2)
                BTGLY1 = sqrtsm2GDY_[i] - (BTGB01 * BTY0 + BTGBBTY_[i] + BTGB12 * BTY2)
                BTGLY2 = sqrtsm2GDY_[j] - (BTGB02 * BTY0 + BTGB12 * BTY1 + BTGBBTY_[j])
                print(CTGDY, BTGLY0)
                BTGB00m1 = BTGB00 - 1.
                BTGB11m1 = sm2Gdiag_[i]
                BTGB22m1 = sm2Gdiag_[j]
                
                CF00 = BTGB11m1 * BTGB22m1 - BTGB12 * BTGB12
                CF01 = -BTGB01 * BTGB22m1 + BTGB12 * BTGB02
                CF02 = BTGB01 * BTGB12 - BTGB11m1 * BTGB02
                CF11 = BTGB00m1 * BTGB22m1 - BTGB02 * BTGB02
                CF12 = -BTGB00m1 * BTGB12 + BTGB01 * BTGB02
                CF22 = BTGB00m1 * BTGB11m1 - BTGB01 * BTGB01
                
                invdeter = 1. / (BTGB00m1 * CF00 + BTGB01 * CF01 + BTGB02 * CF02)
                
                b0 = invdeter * (CF00 * BTGLY0 + CF01 * BTGLY1 + CF02 * BTGLY2) + BTY0
                b1 = invdeter * (CF01 * BTGLY0 + CF11 * BTGLY1 + CF12 * BTGLY2) + BTY1
                b2 = invdeter * (CF02 * BTGLY0 + CF12 * BTGLY1 + CF22 * BTGLY2) + BTY2
                
                t1 = -b0 + sqrtsm2 * b1
                t2 = -b0 + sqrtsm2 * b2
                F0 = GDYi - (Gii * t1 + Gij * t2 + GCi * b0)
                F1 = GDYj - (Gij * t1 + Gjj * t2 + GCj * b0)
                
                results_start.append(F0), results_end.append(F1)
        looppairs(results_start, results_end)
        return np.array(results_start), np.array(results_end)
    
    
    def holdout(self, indices):
        """Computes hold-out predictions for a trained RankRLS
        
        Parameters
        ----------
        indices : list of indices, shape = [n_hsamples]
            list of indices of training examples belonging to the set for which the
            hold-out predictions are calculated. The list can not be empty.

        Returns
        -------
        F : array, shape = [n_hsamples, n_labels]
            holdout predictions
            
        Notes
        -----
    
        The algorithm is a modification of the ones published in [1,2] for the regular RLS method.
        
        References
        ----------
        
        [1] Tapio Pahikkala, Jorma Boberg, and Tapio Salakoski.
        Fast n-Fold Cross-Validation for Regularized Least-Squares.
        Proceedings of the Ninth Scandinavian Conference on Artificial Intelligence,
        83-90, Otamedia Oy, 2006.
        
        [2] Tapio Pahikkala, Hanna Suominen, and Jorma Boberg.
        Efficient cross-validation for kernelized least-squares regression with sparse basis expansions.
        Machine Learning, 87(3):381--407, June 2012.     
        """
        
        indices = array_tools.as_index_list(indices, self.Y.shape[0])
        
        if len(indices) != len(np.unique(indices)):
            raise IndexError('Hold-out can have each index only once.')
        
        Y = self.Y
        m = self.size
        
        evals, V = self.evals, self.svecs
        
        #results = []
        
        C = np.mat(np.zeros((self.size, 3), dtype = np.float64))
        onevec = np.mat(np.ones((self.size, 1), dtype = np.float64))
        for i in range(self.size):
            C[i, 0] = 1.
        
        
        VTY = V.T * Y
        VTC = V.T * onevec
        CTY = onevec.T * Y
        
        holen = len(indices)
        
        newevals = multiply(evals, 1. / ((m - holen) * evals + self.regparam))
        
        R = np.mat(np.zeros((holen, holen + 1), dtype = np.float64))
        for i in range(len(indices)):
            R[i, 0] = -1.
            R[i, i + 1] = sqrt(self.size - float(holen))
        
        Vho = V[indices]
        Vhov = multiply(Vho, newevals)
        Ghoho = Vhov * Vho.T
        GCho = Vhov * VTC
        GBho = Ghoho * R
        for i in range(len(indices)):
            GBho[i, 0] += GCho[i, 0]
        
        CTGC = multiply(VTC.T, newevals) * VTC
        RTGCho = R.T * GCho
        
        BTGB = R.T * Ghoho * R
        for i in range(len(indices) + 1):
            BTGB[i, 0] += RTGCho[i, 0]
            BTGB[0, i] += RTGCho[i, 0]
        BTGB[0, 0] += CTGC[0, 0]
        
        BTY = R.T * Y[indices]
        BTY[0] = BTY[0] + CTY[0]
        
        GDYho = Vhov * (self.size - holen) * VTY
        GLYho = GDYho - GBho * BTY
        
        CTGDY = multiply(VTC.T, newevals) * (self.size - holen) * VTY
        BTGLY = R.T * GDYho - BTGB * BTY
        BTGLY[0] = BTGLY[0] + CTGDY[0]
        
        F = GLYho - GBho * la.inv(-mat(eye(holen + 1)) + BTGB) * BTGLY
        
        #results.append(F)
        #return results
        F = np.squeeze(np.array(F))
        return F
        
    def leave_one_out(self):
        """Computes leave-one-out predictions for a trained RankRLS
        
        Returns
        -------
        F : array, shape = [n_samples, n_labels]
            leave-one-out predictions

        Notes
        -----
    
        Provided for reference, usually you should not call this, but
        rather use leave_pair_out.

        """        
        LOO = mat(zeros((self.size, self.ysize), dtype=float64))
        for i in range(self.size):
            LOO[i,:] = self.holdout([i])
        return LOO
    
    def _reference(self, pairs):
        
        evals, evecs = self.evals, self.svecs
        Y = self.Y
        m = self.size
        
        
        results = []
        
        D = mat(zeros((self.size, 1), dtype=float64))
        C = mat(zeros((self.size, 3), dtype=float64))
        for i in range(self.size):
            D[i, 0] = self.size - 2.
            C[i, 0] = 1.
        lamb = self.regparam
        
        G = evecs * multiply(multiply(evals, 1. / ((m - 2.) * evals + lamb)).T, evecs.T)
        
        
        DY = multiply(D, Y)
        GDY = G * DY
        GC = G * C
        CTG = GC.T
        CT = C.T
        CTGC = CT * GC
        CTY = CT * Y
        CTGDY = CT * GDY
        
        minusI3 = -mat(eye(3))
        for i, j in pairs:
            hoinds = [i, j]
            
            R = mat(zeros((2, 3), dtype=float64))
            R[0, 0] = -1.
            R[1, 0] = -1.
            R[0, 1] = sqrt(self.size - 2.)
            R[1, 2] = sqrt(self.size - 2.)
            
            GBho = GC[hoinds] + G[np.ix_(hoinds, hoinds)] * R
            
            BTGB = CTGC \
                + R.T * GC[hoinds] \
                + CTG[:, hoinds] * R \
                + R.T * G[np.ix_(hoinds, hoinds)] * R
            
            BTY = CTY + R.T * Y[hoinds]
            
            GLYho = GDY[hoinds] - GBho * BTY
            BTGLY = CTGDY + R.T * GDY[hoinds] - BTGB * BTY
            
            F = GLYho - GBho * la.inv(minusI3 + BTGB) * BTGLY
            
            results.append(F)
        return results

class LeavePairOutRankRLS(PredictorInterface):
    """RankRLS: Regularized least-squares ranking. Wrapper code for selecting the
    regularization parameter automatically with leave-pair-out cross-validation.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Data matrix
    Y : {array-like}, shape = [n_samples] or [n_samples, n_labels]
        Training set labels
    kernel : {'LinearKernel', 'GaussianKernel', 'PolynomialKernel', 'PrecomputedKernel', ...}
        kernel function name, imported dynamically from rlscore.kernel
    basis_vectors : {array-like, sparse matrix}, shape = [n_bvectors, n_features], optional
        basis vectors (typically a randomly chosen subset of the training data)
    regparams : {array-like}, shape = [grid_size] (optional)
        regularization parameter values to be tested, default = [2^-15,...,2^15]
        
    Other Parameters
    ----------------
    Typical kernel parameters include:
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
    cv_performances : array, shape = [grid_size]
        leave-pair-out performances for each grid point
    regparam : float
        regparam from grid with best performance
          
    Notes
    -----
    
    Computational complexity of training and model selection:
    m = n_samples, d = n_features, l = n_labels, b = n_bvectors, r = grid_size
    
    O(dm^2 + rlm^2 + rm^3): basic case
    
    O(rlm^2 + rdm^2): Linear Kernel, d < m
    
    O(rlm^2 + rbm^2): Sparse approximation with basis vectors 
    
     
    RankRLS algorithm is described in [1,2].
    
    References
    ----------

    [1] Learning to rank with pairwise regularized least-squares.
    In Thorsten Joachims, Hang Li, Tie-Yan Liu, and ChengXiang Zhai, editors,
    SIGIR 2007 Workshop on Learning to Rank for Information Retrieval, pages 27--33, 2007.
    
    [2] Tapio Pahikkala, Evgeni Tsivtsivadze, Antti Airola, Jouni Jarvinen, and Jorma Boberg.
    An efficient algorithm for learning to rank from preference graphs.
    Machine Learning, 75(1):129-165, 2009.
"""
    
    def __init__(self, X, Y, kernel='LinearKernel', basis_vectors = None, regparams=None, **kwargs):
        if regparams is None:
            grid = [2**x for x in range(-15, 16)]
        else:
            grid = regparams
        learner = GlobalRankRLS(X, Y, grid[0], kernel, basis_vectors, **kwargs)
        crossvalidator = LPOCV(learner)
        self.cv_performances, self.cv_predictions, self.regparam = grid_search(crossvalidator, grid)
        self.predictor = learner.predictor
        
class KfoldRankRLS(PredictorInterface):
    
    """RankRLS: Regularized least-squares ranking. Wrapper code for selecting the
    regularization parameter automatically with K-fold cross-validation.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Data matrix
    Y : {array-like}, shape = [n_samples] or [n_samples, n_labels]
        Training set labels
    kernel : {'LinearKernel', 'GaussianKernel', 'PolynomialKernel', 'PrecomputedKernel', ...}
        kernel function name, imported dynamically from rlscore.kernel
    basis_vectors : {array-like, sparse matrix}, shape = [n_bvectors, n_features], optional
        basis vectors (typically a randomly chosen subset of the training data)
    regparams : {array-like}, shape = [grid_size] (optional)
        regularization parameter values to be tested, default = [2^-15,...,2^15]
        
    Other Parameters
    ----------------
    Typical kernel parameters include:
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
    cv_performances : array, shape = [grid_size]
        K-fold performances for each grid point
    cv_predictions : list of 1D  or 2D arrays, shape = [grid_size, n_folds]
        predictions for each fold, shapes [fold_size] or [fold_size, n_labels]
    regparam : float
        regparam from grid with best performance
           
    Notes
    -----
     
    RankRLS algorithm is described in [1,2]. 
    
    [1] Tapio Pahikkala, Evgeni Tsivtsivadze, Antti Airola, Jorma Boberg and Tapio Salakoski
    Learning to rank with pairwise regularized least-squares.
    In Thorsten Joachims, Hang Li, Tie-Yan Liu, and ChengXiang Zhai, editors,
    SIGIR 2007 Workshop on Learning to Rank for Information Retrieval, pages 27--33, 2007.
    
    [2] Tapio Pahikkala, Evgeni Tsivtsivadze, Antti Airola, Jouni Jarvinen, and Jorma Boberg.
    An efficient algorithm for learning to rank from preference graphs.
    Machine Learning, 75(1):129-165, 2009.
"""
    
    def __init__(self, X, Y, folds, kernel='LinearKernel', basis_vectors = None, regparams=None, measure=None, save_predictions = False, **kwargs):
        if regparams is None:
            grid = [2**x for x in range(-15, 15)]
        else:
            grid = regparams
        if measure is None:
            self.measure = cindex
        else:
            self.measure = measure
        learner = GlobalRankRLS(X, Y, grid[0], kernel, basis_vectors, **kwargs)
        crossvalidator = NfoldCV(learner, measure, folds)
        self.cv_performances, self.cv_predictions, self.regparam = grid_search(crossvalidator, grid)
        self.predictor = learner.predictor


                
        
        
    

