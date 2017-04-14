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

from numpy import identity, multiply, mat, sum
import numpy.linalg as la
from rlscore.utilities import array_tools 
from rlscore.utilities import adapter
from . import _rls

from rlscore.measure.measure_utilities import UndefinedPerformance
from rlscore.measure import sqerror
from rlscore.measure import cindex
import numpy as np
from rlscore.predictor import PredictorInterface
from rlscore.utilities.cross_validation import grid_search

class RLS(PredictorInterface):
    """Regularized least-squares regression/classification

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
    predictor: {LinearPredictor, KernelPredictor}
        trained predictor
                  
    Notes
    -----
    
    Computational complexity of training:
    m = n_samples, d = n_features, l = n_labels, b = n_bvectors
    
    O(m^3 + dm^2 + lm^2): basic case
    
    O(md^2 +lmd): Linear Kernel, d < m
    
    O(mb^2 +lmb): Sparse approximation with basis vectors 
     
    Basic information about RLS, and a description of the fast leave-one-out method
    can be found in [1]. The efficient K-fold cross-validation algorithm implemented in
    the method holdout is based on results in [2] and [3]. The leave-pair-out cross-validation
    algorithm implemented in leave_pair_out is a modification of the method described
    in [4] , its use for AUC-estimation has been analyzed in [5].
    
    References
    ----------
    [1] Ryan Rifkin, Ross Lippert. Notes on Regularized Least Squares.
    Technical Report, MIT, 2007.
    
    [2] Tapio Pahikkala, Jorma Boberg, and Tapio Salakoski.
    Fast n-Fold Cross-Validation for Regularized Least-Squares.
    Proceedings of the Ninth Scandinavian Conference on Artificial Intelligence,
    83-90, Otamedia Oy, 2006.
    
    [3] Tapio Pahikkala, Hanna Suominen, and Jorma Boberg.
    Efficient cross-validation for kernelized least-squares regression with sparse basis expansions.
    Machine Learning, 87(3):381--407, June 2012. 
    
    [4] Tapio Pahikkala, Antti Airola, Jorma Boberg, and Tapio Salakoski.
    Exact and efficient leave-pair-out cross-validation for ranking RLS.
    In Proceedings of the 2nd International and Interdisciplinary Conference
    on Adaptive Knowledge Representation and Reasoning (AKRR'08), pages 1-8,
    Espoo, Finland, 2008.
    
    [5] Antti Airola, Tapio Pahikkala, Willem Waegeman, Bernard De Baets, and Tapio Salakoski
    An experimental comparison of cross-validation techniques for estimating the area under the ROC curve.
    Computational Statistics & Data Analysis, 55(4):1828-1844, April 2011.
    """

    def __init__(self, X, Y, regparam = 1.0, kernel='LinearKernel', basis_vectors = None, **kwargs):
        self.Y = array_tools.as_2d_array(Y)
        if X.shape[0] != Y.shape[0]:
            raise Exception("First dimension of X and Y must be the same")
        if basis_vectors is not None:
            if X.shape[1] != basis_vectors.shape[1]:
                raise Exception("Number of columns for X and basis_vectors must be the same")
        kwargs['X'] = X
        kwargs['kernel'] = kernel
        if basis_vectors is not None:
            kwargs['basis_vectors'] = basis_vectors
        self.svdad = adapter.createSVDAdapter(**kwargs)
        self.regparam = regparam
        self.svals = np.mat(self.svdad.svals)
        self.svecs = np.mat(self.svdad.rsvecs)
        self.size = self.Y.shape[0]
        self.solve(self.regparam)   
   
    def solve(self, regparam=1.0):
        """Re-trains RLS for the given regparam
               
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
        
        References
        ----------
        
        [1] Ryan Rifkin, Ross Lippert.
        Notes on Regularized Least Squares.
        Technical Report, MIT, 2007.        
        """
        
        if not hasattr(self, "svecsTY"):
            self.svecsTY = self.svecs.T * self.Y
        
            #Eigenvalues of the kernel matrix
            self.evals = multiply(self.svals, self.svals)
        
        self.newevals = 1. / (self.evals + regparam)
        self.regparam = regparam
        self.A = self.svecs * multiply(self.newevals.T, self.svecsTY)
        self.predictor = self.svdad.createModel(self)
    
    
    def holdout(self, indices):
        """Computes hold-out predictions
        
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
        
        Computational complexity of holdout:
        m = n_samples, d = n_features, l = n_labels, b = n_bvectors, h=n_hsamples
        
        O(h^3 + lmh): basic case
        
        O(min(h^3 + lh^2, d^3 + ld^2) +ldh): Linear Kernel, d < m
        
        O(min(h^3 + lh^2, b^3 + lb^2) +lbh): Sparse approximation with basis vectors 
        
        The fast holdout algorithm is based on results presented in [1,2]. However, the removal of basis vectors decribed in [2] is currently not implemented.
            
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
        
        bevals = multiply(self.evals, self.newevals)
        A = self.svecs[indices]
        right = self.svecsTY - A.T * self.Y[indices] #O(hrl)
        RQY = A * multiply(bevals.T, right) #O(hrl)
        B = multiply(bevals.T, A.T)
        if len(indices) <= A.shape[1]: #h < r
            I = mat(identity(len(indices)))
            result = la.inv(I - A * B) * RQY #O(h^3 + h^2 * l)
        else: #h > r
            I = mat(identity(A.shape[1]))
            result = RQY - A * (la.inv(B * A - I) * (B * RQY)) #O(r^3 + r^2 * l + h * r * l)
        return np.squeeze(np.array(result))
    
    
    def leave_one_out(self):
        """Computes leave-one-out predictions
        
        Returns
        -------
        F : array, shape = [n_samples, n_labels]
            leave-one-out predictions

        Notes
        -----
    
        Computational complexity of leave-one-out:
        m = n_samples, d = n_features, l = n_labels, b = n_bvectors
        
        O(lm^2): basic case
        
        O(ldm): Linear Kernel, d < m
        
        O(lbm): Sparse approximation with basis vectors 
        
        Implements the classical leave-one-out algorithm described for example in [1].
        
        References
        ----------
                    
        [1] Ryan Rifkin, Ross Lippert.
        Notes on Regularized Least Squares.
        Technical Report, MIT, 2007.

        """
        bevals = multiply(self.evals, self.newevals)
        svecsm = multiply(bevals, self.svecs)
        RQR = sum(multiply(self.svecs, svecsm), axis = 1)
        LOO_ek = (1. / (1. - RQR))
        LOO = multiply(LOO_ek, self.svecs * (svecsm.T * self.Y)) - multiply(LOO_ek, multiply(RQR, self.Y))
        return np.squeeze(np.array(LOO))
    
    
    def leave_pair_out(self, pairs_start_inds, pairs_end_inds):
        
        """Computes leave-pair-out predictions
        
        Parameters
        ----------
        pairs_start_inds : list of indices, shape = [n_pairs]
            list of indices from range [0, n_samples-1]
        pairs_end_inds : list of indices, shape = [n_pairs]
            list of indices from range [0, n_samples-1]
        
        Returns
        -------
        P1 : array, shape = [n_pairs, n_labels]
            holdout predictions for pairs_start_inds
        P2 : array, shape = [n_pairs, n_labels]
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
        
        The algorithm is an adaptation of the method published originally in [1]. The use of
        leave-pair-out cross-validation for AUC estimation has been analyzed in [2].

        References
        ---------- 
        
        [1] Tapio Pahikkala, Antti Airola, Jorma Boberg, and Tapio Salakoski.
        Exact and efficient leave-pair-out cross-validation for ranking RLS.
        In Proceedings of the 2nd International and Interdisciplinary Conference
        on Adaptive Knowledge Representation and Reasoning (AKRR'08), pages 1-8,
        Espoo, Finland, 2008.
        
        [2] Antti Airola, Tapio Pahikkala, Willem Waegeman, Bernard De Baets, and Tapio Salakoski.
        An experimental comparison of cross-validation techniques for estimating the area under the ROC curve.
        Computational Statistics & Data Analysis, 55(4):1828--1844, April 2011.
        """
        
        pairs_start_inds = array_tools.as_index_list(pairs_start_inds, self.Y.shape[0])
        pairs_end_inds = array_tools.as_index_list(pairs_end_inds, self.Y.shape[0])
        pairslen = len(pairs_start_inds)
        if not len(pairs_start_inds) == len(pairs_end_inds):
            raise Exception("Incorrect arguments: lengths of pairs_start_inds and pairs_end_inds do no match")
        
        bevals = multiply(self.evals, self.newevals)
        svecsbevals = multiply(self.svecs, bevals)
        hatmatrixdiagonal = np.squeeze(np.array(np.sum(np.multiply(self.svecs, svecsbevals), axis = 1)))
        svecsbevalssvecsTY = svecsbevals * self.svecsTY
        results_first = np.zeros((pairslen, self.Y.shape[1]))
        results_second = np.zeros((pairslen, self.Y.shape[1]))
        _rls.leave_pair_out(pairslen,
                                                     pairs_start_inds,
                                                     pairs_end_inds,
                                                     self.Y.shape[1],
                                                     self.Y,
                                                     self.svecs,
                                                     np.atleast_1d(np.squeeze(np.array(bevals))),
                                                     self.svecs.shape[1],
                                                     hatmatrixdiagonal,
                                                     svecsbevalssvecsTY,
                                                     results_first,
                                                     results_second)
        return np.squeeze(results_first), np.squeeze(results_second)


class LeaveOneOutRLS(PredictorInterface):
    
    """Regularized least-squares regression/classification. Wrapper code that selects
    regularization parameter automatically based on leave-one-out cross-validation.

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
    measure : function(Y, P) (optional)
        a performance measure from rlscore.measure used for model selection,
        default sqerror (mean squared error)

        
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
        leave-one-out performances for each grid point
    cv_predictions : array, shape = [grid_size, n_samples] or [grid_size, n_samples, n_labels]
        leave-one-out predictions
    regparam : float
        regparam from grid with best performance
                  
    Notes
    -----
    
    Computational complexity of training (model selection is basically free due to fast regularization and leave-one-out):
    m = n_samples, d = n_features, l = n_labels, b = n_bvectors, r=grid_size
    
    O(m^3 + dm^2 + rlm^2): basic case
    
    O(md^2 + rlmd): Linear Kernel, d < m
    
    O(mb^2 + rlmb): Sparse approximation with basis vectors 
     
    Basic information about RLS, and a description of the fast leave-one-out method
    can be found in [1]. 

    References
    ---------- 
               
    [1] Ryan Rifkin, Ross Lippert.
    Notes on Regularized Least Squares.
    Technical Report, MIT, 2007.
    """
    
    def __init__(self, X, Y, kernel='LinearKernel', basis_vectors = None, regparams=None, measure=None, **kwargs):
        if regparams is None:
            grid = [2**x for x in range(-15, 16)]
        else:
            grid = regparams
        if measure is None:
            measure = sqerror
        learner = RLS(X, Y, grid[0], kernel, basis_vectors, **kwargs)
        crossvalidator = LOOCV(learner, measure)
        self.cv_performances, self.cv_predictions, self.regparam = grid_search(crossvalidator, grid)
        self.cv_predictions = np.array(self.cv_predictions)
        self.predictor = learner.predictor
            
class KfoldRLS(PredictorInterface):
    
    """Regularized least-squares regression/classification. Wrapper code that selects
    regularization parameter automatically based on K-fold cross-validation.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Data matrix
    Y : {array-like}, shape = [n_samples] or [n_samples, n_labels]
        Training set labels
    folds : list of index lists, shape = [n_folds]
        Each list within the folds list contains the indices of samples in one fold, indices
        must be from range [0,n_samples-1]
    kernel : {'LinearKernel', 'GaussianKernel', 'PolynomialKernel', 'PrecomputedKernel', ...}
        kernel function name, imported dynamically from rlscore.kernel
    basis_vectors : {array-like, sparse matrix}, shape = [n_bvectors, n_features], optional
        basis vectors (typically a randomly chosen subset of the training data)
    regparams : {array-like}, shape = [grid_size] (optional)
        regularization parameter values to be tested, default = [2^-15,...,2^15]
    measure : function(Y, P) (optional)
        a performance measure from rlscore.measure used for model selection,
        default sqerror (squared error)

        
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
    
    Uses fast solve and holdout algorithms, complexity depends on the sizes of the folds. Complexity
    when using K-fold cross-validation is:
    
    m = n_samples, d = n_features, l = n_labels, b = n_bvectors, r=grid_size, k = n_folds  

    O(m^3 + dm^2 + r*(m^3/k^2 + lm^2)): basic case
    
    O(md^2 + r*(min(m^3/k^2 + lm^2/k, kd^3 + kld^2) + ldm)): Linear Kernel, d < m
    
    O(mb^2 + r*(min(m^3/k^2 + lm^2/k, kb^3 + klb^2) + lbm)): Sparse approximation with basis vectors


    Basic information about RLS can be found in [1]. The K-fold algorithm is based on results published
    in [2] and [3].

    References
    ---------- 
               
    [1] Ryan Rifkin, Ross Lippert.
    Notes on Regularized Least Squares.
    Technical Report, MIT, 2007.
    
    [2] Tapio Pahikkala, Jorma Boberg, and Tapio Salakoski.
    Fast n-Fold Cross-Validation for Regularized Least-Squares.
    Proceedings of the Ninth Scandinavian Conference on Artificial Intelligence,
    83-90, Otamedia Oy, 2006.
        
    [3] Tapio Pahikkala, Hanna Suominen, and Jorma Boberg.
    Efficient cross-validation for kernelized least-squares regression with sparse basis expansions.
    Machine Learning, 87(3):381--407, June 2012.   
    """
    
    def __init__(self, X, Y, folds, kernel='LinearKernel', basis_vectors = None, regparams=None, measure=None, save_predictions = False, **kwargs):
        if regparams is None:
            grid = [2**x for x in range(-15, 16)]
        else:
            grid = regparams
        if measure is None:
            self.measure = sqerror
        else:
            self.measure = measure
        learner = RLS(X, Y, grid[0], kernel, basis_vectors, **kwargs)
        crossvalidator = NfoldCV(learner, measure, folds)
        self.cv_performances, self.cv_predictions, self.regparam = grid_search(crossvalidator, grid)
        self.predictor = learner.predictor
        
class LeavePairOutRLS(PredictorInterface):
    
    """Regularized least-squares regression/classification. Wrapper code that selects
    regularization parameter automatically based on ranking accuracy (area under ROC curve
    for binary classification tasks) in leave-pair-out cross-validation.

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
    measure : function(Y, P) (optional)
        a performance measure from rlscore.measure used for model selection,
        default sqerror (squared error)

        
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
    
    O(rlm^2 + dm^2 + rm^3): basic case
    
    O(rlm^2 + rdm^2): Linear Kernel, d < m
    
    O(rlm^2 + rbm^2): Sparse approximation with basis vectors 
     
    Basic information about RLS can be found in [1]. The leave-pair-out algorithm
    is an adaptation of the method published in [2]. The use of leave-pair-out
    cross-validation for AUC estimation has been analyzed in [3].
    
    References
    ----------    
        
    [1] Ryan Rifkin, Ross Lippert.
    Notes on Regularized Least Squares.
    Technical Report, MIT, 2007.
    
    [2] Tapio Pahikkala, Antti Airola, Jorma Boberg, and Tapio Salakoski.
    Exact and efficient leave-pair-out cross-validation for ranking RLS.
    In Proceedings of the 2nd International and Interdisciplinary Conference
    on Adaptive Knowledge Representation and Reasoning (AKRR'08), pages 1-8,
    Espoo, Finland, 2008.
        
    [3] Antti Airola, Tapio Pahikkala, Willem Waegeman, Bernard De Baets, and Tapio Salakoski.
    An experimental comparison of cross-validation techniques for estimating the area under the ROC curve.
    Computational Statistics & Data Analysis, 55(4):1828--1844, April 2011. 
    """
    
    def __init__(self, X, Y, kernel='LinearKernel', basis_vectors = None, regparams=None, **kwargs):
        if regparams is None:
            grid = [2**x for x in range(-15, 16)]
        else:
            grid = regparams
        learner = RLS(X, Y, grid[0], kernel, basis_vectors, **kwargs)
        crossvalidator = LPOCV(learner)
        self.cv_performances, self.cv_predictions, self.regparam = grid_search(crossvalidator, grid)
        self.predictor = learner.predictor


class LOOCV(object):
    
    def __init__(self, learner, measure):
        self.rls = learner
        self.measure = measure
        
    def cv(self, regparam):
        self.rls.solve(regparam)
        Y = self.rls.Y
        P = self.rls.leave_one_out()
        perf = self.measure(Y, P)
        return perf, P

class NfoldCV(object):
    
    def __init__(self, learner, measure, folds):
        self.rls = learner
        if measure is None:
            self.measure = sqerror
        else:
            self.measure = measure
        self.folds = folds
        
    def cv(self, regparam):
        rls = self.rls
        folds = self.folds
        measure = self.measure
        rls.solve(regparam)
        Y = rls.Y
        performances = []
        P_all = []
        for fold in folds:
            P = rls.holdout(fold)
            P_all.append(P)
            try:
                performance = measure(Y[fold], P)
                performances.append(performance)
            except UndefinedPerformance as e:
                pass #No warning printed
            #performance = measure_utilities.aggregate(performances)
        if len(performances) > 0:
            performance = np.mean(performances)
        else:
            raise UndefinedPerformance("Performance undefined for all folds")
        return performance, P_all

class LPOCV(object):
    
    def __init__(self, learner):
        self.rls = learner
        self.measure = cindex

    def cv_old(self, regparam):
        rls = self.rls
        rls.solve(regparam)
        Y = rls.Y
        aucs = []
        for k in range(Y.shape[1]):
            pairs_start_inds, pairs_end_inds = [], []
            for i in range(Y.shape[0] - 1):
                for j in range(i + 1, Y.shape[0]):
                    if Y[i,k] > Y[j,k]:
                        pairs_start_inds.append(i)
                        pairs_end_inds.append(j)
                    elif Y[i,k] < Y[j,k]:
                        pairs_start_inds.append(j)
                        pairs_end_inds.append(i)
            if len(pairs_start_inds) == 0:
                raise UndefinedPerformance("Leave-pair-out undefined, all labels same for output %d" %k)
            pred_start, pred_end = rls.leave_pair_out(np.array(pairs_start_inds), np.array(pairs_end_inds))
            pred_start = array_tools.as_2d_array(pred_start)
            pred_end = array_tools.as_2d_array(pred_end)
            auc = 0.
            for h in range(len(pred_start)):
                if pred_start[h,k] > pred_end[h,k]:
                    auc += 1.
                elif pred_start[h,k] == pred_end[h,k]:
                    auc += 0.5
            auc /= len(pairs_start_inds)
            aucs.append(auc)
        auc = np.mean(aucs)
        return auc, None
    
    def cv(self, regparam):
        rls = self.rls
        rls.solve(regparam)
        Y = rls.Y
        #Union of all pairs for which predictions are needed
        all_pairs = set([])
        for k in range(Y.shape[1]):
            pairs = []
            for i in range(Y.shape[0] - 1):
                for j in range(i + 1, Y.shape[0]):
                    if Y[i,k] != Y[j,k]:
                        pairs.append((i,j))
            #If all labels for some column are same, ranking accuracy is undefined
            if len(pairs) == 0:
                raise UndefinedPerformance("Leave-pair-out undefined, all labels same for output %d" %k)
            all_pairs.update(pairs)
        all_start_inds = [x[0] for x in all_pairs]
        all_end_inds = [x[1] for x in all_pairs]
        #Compute leave-pair-out predictions for all pairs
        all_start_inds = np.array(all_start_inds)
        all_end_inds = np.array(all_end_inds)
        pred_start, pred_end = rls.leave_pair_out(all_start_inds, all_end_inds)
        pred_start = array_tools.as_2d_array(pred_start)
        pred_end = array_tools.as_2d_array(pred_end)
        pair_dict = dict(zip(all_pairs, list(range(pred_start.shape[0]))))
        aucs = []
        #compute auc/ranking accuracy for each column of Y separately
        for k in range(Y.shape[1]):
            comparisons = []
            #1 if the true and predicted agree, 0 if disagree, 0.5 if predictions tied
            for i in range(Y.shape[0] - 1):
                for j in range(i + 1, Y.shape[0]):
                    if Y[i,k] > Y[j,k]:
                        ind = pair_dict[(i,j)]
                        if pred_start[ind,k] > pred_end[ind,k]:
                            comparisons.append(1.)
                        elif pred_start[ind,k] == pred_end[ind,k]:
                            comparisons.append(0.5)
                        else:
                            comparisons.append(0.)
                    elif Y[i,k] < Y[j,k]:
                        ind = pair_dict[(i,j)]
                        if pred_start[ind,k] < pred_end[ind,k]:
                            comparisons.append(1.)
                        elif pred_start[ind,k] == pred_end[ind,k]:
                            comparisons.append(0.5)
                        else:
                            comparisons.append(0.)
            auc = np.mean(comparisons)
            aucs.append(auc)
        #Take the mean of all columnwise aucs
        auc = np.mean(aucs)
        return auc, None
                
