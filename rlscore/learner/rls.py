
from numpy import identity, multiply, mat, sum
import numpy.linalg as la
from rlscore.utilities import array_tools
from rlscore.utilities import creators
import cython_pairwise_cv_for_rls

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
    X: {array-like, sparse matrix}, shape = [n_samples, n_features]
        Data matrix
        
    Y: {array-like}, shape = [n_samples] or [n_samples, n_labels]
        Training set labels
        
    regparam: float, optional
        regularization parameter, regparam > 0 (default=1.0)
        
    kernel: {'LinearKernel', 'GaussianKernel', 'PolynomialKernel', 'PrecomputedKernel', ...}
        kernel function name, imported dynamically from rlscore.kernel
        
    basis_vectors: {array-like, sparse matrix}, shape = [n_bvectors, n_features], optional
        basis vectors (typically a randomly chosen subset of the training data)
        
    Other Parameters
    ----------------
    bias: float, optional
        LinearKernel: the model is w*x + bias*w0, (default=1.0)
        
    gamma: float, optional
        GaussianKernel: k(xi,xj) = e^(-gamma*<xi-xj,xi-xj>) (default=1.0)
        PolynomialKernel: k(xi,xj) = (gamma * <xi, xj> + coef0)**degree (default=1.0)
        
    degree: float, optional
        PolynomialKernel: k(xi,xj) = (gamma * <xi, xj> + coef0)**degree (default=1.0) 
               
    coef0: float, optional
        PolynomialKernel: k(xi,xj) = (gamma * <xi, xj> + coef0)**degree (default=0.)
        
    degree: int, optional
        PolynomialKernel: k(xi,xj) = (gamma * <xi, xj> + coef0)**degree (default=2)
                  
    Notes
    -----
    
    Computational complexity of training:
    m = n_samples, d = n_features, l = n_labels, b = n_bvectors
    
    O(m^3 + lm^2): basic case
    O(md^2 +lmd): Linear Kernel, d < m
    O(mb^2 +lmb): Sparse approximation with basis vectors 
     
    Basic information about RLS, and a description of the fast leave-one-out method
    can be found in [1]_. The efficient K-fold cross-validation algorithm implemented in
    the method holdout is based on results in [2]_ and [3]_. The leave-pair-out cross-validation
    algorithm implemented in leave_pairs_out is a modification of the method described
    in [4]_ , its use for AUC-estimation has been analyzed in [5]_.
    
    References
    ----------
    .. [1] Ryan Rifkin, Ross Lippert. Notes on Regularized Least Squares
    Technical Report, MIT, 2007.
    
    .. [2] Tapio Pahikkala, Jorma Boberg, and Tapio Salakoski.
    Fast n-Fold Cross-Validation for Regularized Least-Squares.
    Proceedings of the Ninth Scandinavian Conference on Artificial Intelligence,
    83-90, Otamedia Oy, 2006.
    
    .. [3] Tapio Pahikkala, Hanna Suominen, and Jorma Boberg.
    Efficient cross-validation for kernelized least-squares regression with sparse basis expansions.
    Machine Learning, 87(3):381--407, June 2012. 
    
    .. [4] Tapio Pahikkala, Antti Airola, Jorma Boberg, and Tapio Salakoski.
    Exact and efficient leave-pair-out cross-validation for ranking RLS.
    In Proceedings of the 2nd International and Interdisciplinary Conference
    on Adaptive Knowledge Representation and Reasoning (AKRR'08), pages 1-8,
    Espoo, Finland, 2008.
    
    .. [5] Antti Airola, Tapio Pahikkala, Willem Waegeman, Bernard De Baets, and Tapio Salakoski
    An experimental comparison of cross-validation techniques for estimating the area under the ROC curve.
    Computational Statistics & Data Analysis, 55(4):1828-1844, April 2011.
    """

    def __init__(self, X, Y, regparam = 1.0, kernel='LinearKernel', basis_vectors = None, **kwargs):
        kwargs['X'] = X
        kwargs['kernel'] = kernel
        if basis_vectors != None:
            kwargs['basis_vectors'] = basis_vectors
        self.svdad = creators.createSVDAdapter(**kwargs)
        self.Y = array_tools.as_labelmatrix(Y)
        self.regparam = regparam
        self.svals = self.svdad.svals
        self.svecs = self.svdad.rsvecs
        self.size = self.Y.shape[0]
        self.solve(self.regparam)   
   
    def solve(self, regparam=1.0):
        """Re-trains RLS for the given regparam.
               
        Parameters
        ----------
        regparam: float, optional
            regularization parameter, regparam > 0 (default=1.0)
            
        Notes
        -----
    
        Computational complexity of re-training:
        m = n_samples, d = n_features, l = n_labels, b = n_bvectors
        
        O(lm^2): basic case
        
        O(ld^2): Linear Kernel, d < m
        
        O(lb^2): Sparse approximation with basis vectors 
        
        See:
        Ryan Rifkin, Ross Lippert.
        Notes on Regularized Least Squares
        Technical Report, MIT, 2007.        
        """
        
        if not hasattr(self, "svecsTY"):
            #print self.svals.shape
            self.svecsTY = self.svecs.T * self.Y
        
            #Eigenvalues of the kernel matrix
            self.evals = multiply(self.svals, self.svals)
        
        self.newevals = 1. / (self.evals + regparam)
        self.regparam = regparam
        self.A = self.svecs * multiply(self.newevals.T, self.svecsTY)
        #self.results["model"] = self.getModel()
        #if self.U == None:
        #    pass
            #Dual RLS
            #self.A = self.svecs * multiply(self.newevals.T, self.svecsTY)
        #else:
            #Primal RLS
            #bevals = multiply(self.svals, self.newevals)
            #self.A = self.U.T * multiply(bevals.T, self.svecsTY)
        #    self.A = self.U.T * multiply(self.svals.T, self.svecs.T * self.A)
        self.predictor = self.svdad.createModel(self)
    
    
    def holdout(self, indices):
        """Computes hold-out predictions for a trained RLS.
        
        Parameters
        ----------
        indices: list of indices, shape = [n_hsamples]
            list of indices of training examples belonging to the set for which the
            hold-out predictions are calculated. The list can not be empty.

        Returns
        -------
        F : array, shape = [n_hsamples, n_labels]
            holdout predictions
            
        Notes
        -----
        
        The algorithm is based on results published in:
        
        Tapio Pahikkala, Jorma Boberg, and Tapio Salakoski.
        Fast n-Fold Cross-Validation for Regularized Least-Squares.
        Proceedings of the Ninth Scandinavian Conference on Artificial Intelligence,
        83-90, Otamedia Oy, 2006.
        
        Tapio Pahikkala, Hanna Suominen, and Jorma Boberg.
        Efficient cross-validation for kernelized least-squares regression with sparse basis expansions.
        Machine Learning, 87(3):381--407, June 2012.     
        """
        
        if len(indices) == 0:
            raise Exception('Hold-out predictions can not be computed for an empty hold-out set.')
        
        if len(indices) != len(set(indices)):
            raise Exception('Hold-out can have each index only once.')
        
        bevals = multiply(self.evals, self.newevals)
        A = self.svecs[indices]
        right = self.svecsTY - A.T * self.Y[indices]
        RQY = A * multiply(bevals.T, right)
        B = multiply(bevals.T, A.T)
        if len(indices) <= A.shape[1]:
            I = mat(identity(len(indices)))
            result = la.solve(I - A * B, RQY)
        else:
            I = mat(identity(A.shape[1]))
            result = RQY - A * (la.inv(B * A - I) * (B * RQY))
        return np.array(result)
    
    
    def leave_one_out(self):
        """Computes leave-one-out predictions for a trained RLS.
        
        Returns
        -------
        F : array, shape = [n_samples, n_labels]
            leave-one-out predictions

        Notes
        -----
    
        Computational complexity of leave-one-out:
        Computational complexity of re-training:
        m = n_samples, d = n_features, l = n_labels, b = n_bvectors
        
        O(lm^2): basic case
        
        O(ld^2): Linear Kernel, d < m
        
        O(lb^2): Sparse approximation with basis vectors 
        
        Implements the classical leave-one-out algorithm described for example in:            
        Ryan Rifkin, Ross Lippert.
        Notes on Regularized Least Squares
        Technical Report, MIT, 2007.

        """
        bevals = multiply(self.evals, self.newevals)
        #rightall = multiply(bevals.T, self.svecs.T * self.Y)
        '''for i in range(self.size):
            RV = self.svecs[i]
            RVm = multiply(bevals, RV)
            right = rightall - RVm.T * self.Y[i]
            RQY = RV * right
            RQRT = RV * RVm.T
            LOO[i] = (1. / (1. - RQRT)) * RQY'''
        svecsm = multiply(bevals, self.svecs)
        #print rightall.shape, svecsm.shape, self.Y.shape
        #right = svecsm.T * self.Y - multiply(svecsm, self.Y).T
        RQR = sum(multiply(self.svecs, svecsm), axis = 1)
        #RQY = sum(multiply(self.svecs.T, right), axis = 0)
        #RQY = sum(multiply(self.svecs.T, svecsm.T * self.Y), axis = 0) - sum(multiply(RQRT.T, self.Y), axis = 1).T
        #RQY = self.svecs * (svecsm.T * self.Y) - sum(multiply(RQR, self.Y), axis = 1)
        LOO_ek = (1. / (1. - RQR))
        #LOO = multiply(LOO_ek, RQY)
        #print LOO_ek.shape, (self.svecs * (svecsm.T * self.Y)).shape, RQR.shape, self.Y.shape
        LOO = multiply(LOO_ek, self.svecs * (svecsm.T * self.Y)) - multiply(LOO_ek, multiply(RQR, self.Y))
        return np.array(LOO)
    
    
    def leave_pairs_out(self, pairs_start_inds, pairs_end_inds):
        
        """Computes leave-pair-out predictions for a trained RLS.
        
        Parameters
        ----------
        pairs_start_inds: list of indices, shape = [n_pairs]
            list of indices from range [0, n_samples-1]
        pairs_end_inds: list of indices, shape = [n_pairs]
            list of indices from range [0, n_samples-1]
        
        Returns
        -------
        P1 : array, shape = [n_pairs, n_labels]
            holdout predictions for pairs_start_inds
        P2: array, shape = [n_pairs, n_labels]
            holdout predictions for pairs_end_inds
            
        Notes
        -----
    
        Computes the leave-pair-out cross-validation predicitons, where each (i,j) pair with
        i= pair_start_inds[k] and j = pairs_end_inds[k] is left out in turn.
        
        When estimating area under ROC curve with leave-pair-out, one should leave out all
        positive-negative pairs, while for estimating the general ranking error one should
        leave out all pairs with different labels.
        
        Computational complexity of holdout:
        m = n_samples, l=n_labels
        O(m^3 + lm^2) basic case
        O(dm^2 + lm^2) linear, if d<m
        O(bm^2 + lm^2) sparse approximation
        
        The algorithm is an adaptation of the method published originally in [1]_. The use of
        leave-pair-out cross-validation for AUC estimation has been analyzed in [2]_.

        References
        ---------- 
        
        .. [1] Tapio Pahikkala, Antti Airola, Jorma Boberg, and Tapio Salakoski.
        Exact and efficient leave-pair-out cross-validation for ranking RLS.
        In Proceedings of the 2nd International and Interdisciplinary Conference
        on Adaptive Knowledge Representation and Reasoning (AKRR'08), pages 1-8,
        Espoo, Finland, 2008.
        
        .. [2] Antti Airola, Tapio Pahikkala, Willem Waegeman, Bernard De Baets, and Tapio Salakoski.
        An experimental comparison of cross-validation techniques for estimating the area under the ROC curve.
        Computational Statistics & Data Analysis, 55(4):1828--1844, April 2011.
        """
        
        pairslen = len(pairs_start_inds)
        assert len(pairs_end_inds) == pairslen
        
        bevals = multiply(self.evals, self.newevals)
        svecsbevals = multiply(self.svecs, bevals)
        hatmatrixdiagonal = np.squeeze(np.array(np.sum(np.multiply(self.svecs, svecsbevals), axis = 1)))
        #svecsbevalssvecsT = svecsbevals * self.svecs.T
        svecsbevalssvecsTY = svecsbevals * self.svecsTY
        results_first = np.zeros((pairslen, self.Y.shape[1]))
        results_second = np.zeros((pairslen, self.Y.shape[1]))
        cython_pairwise_cv_for_rls.leave_pairs_out(pairslen,
                                                     pairs_start_inds,
                                                     pairs_end_inds,
                                                     self.Y.shape[1],
                                                     self.Y,
                                                     #svecsbevalssvecsT,
                                                     self.svecs,
                                                     np.squeeze(np.array(bevals)),
                                                     self.svecs.shape[1],
                                                     hatmatrixdiagonal,
                                                     svecsbevalssvecsTY,
                                                     results_first,
                                                     results_second)
        return np.squeeze(results_first), np.squeeze(results_second)
        
        '''
        results = []
        
        for i, j in pairs:
            indices = [i, j]
            RQY = svecsbevalssvecsTY[indices] - svecsbevalssvecsT[np.ix_(indices, indices)] * self.Y[indices]
            
            #Invert a symmetric 2x2 matrix
            a, b, d = 1. - svecsbevalssvecsT[i, i], - svecsbevalssvecsT[i, j], 1. - svecsbevalssvecsT[j, j]
            det = 1. / (a * d - b * b)
            ia, ib, id = det * d, - det * b, det * a
            
            lpo_i = ia * RQY[0] + ib * RQY[1]
            lpo_j = ib * RQY[0] + id * RQY[1]
            #result = la.solve(I - A * B, RQY)
            results.append([lpo_i, lpo_j])
        return np.array(results)'''


class LeaveOneOutRLS(PredictorInterface):
    
    """Regularized least-squares regression/classification. Wrapper code that selects
    regularization parameter automatically based on leave-one-out cross-validation.

    Parameters
    ----------
    X: {array-like, sparse matrix}, shape = [n_samples, n_features]
        Data matrix
    Y: {array-like}, shape = [n_samples] or [n_samples, n_labels]
        Training set labels
    kernel: {'LinearKernel', 'GaussianKernel', 'PolynomialKernel', 'PrecomputedKernel', ...}
        kernel function name, imported dynamically from rlscore.kernel
    basis_vectors: {array-like, sparse matrix}, shape = [n_bvectors, n_features], optional
        basis vectors (typically a randomly chosen subset of the training data)
    regparams: {array-like}, shape = [grid_size] (optional)
        regularization parameter values to be tested, default = [2^-15,...,2^15]
    measure: function(Y, P) (optional)
        a performance measure from rlscore.measure used for model selection,
        default sqerror (squared error)

        
    Other Parameters
    ----------------
    Typical kernel parameters include:
    bias: float, optional
        LinearKernel: the model is w*x + bias*w0, (default=1.0)
    gamma: float, optional
        GaussianKernel: k(xi,xj) = e^(-gamma*<xi-xj,xi-xj>) (default=1.0)
        PolynomialKernel: k(xi,xj) = (gamma * <xi, xj> + coef0)**degree (default=1.0)
    degree: float, optional
        PolynomialKernel: k(xi,xj) = (gamma * <xi, xj> + coef0)**degree (default=1.0)        
    coef0: float, optional
        PolynomialKernel: k(xi,xj) = (gamma * <xi, xj> + coef0)**degree (default=0.)
    degree: int, optional
        PolynomialKernel: k(xi,xj) = (gamma * <xi, xj> + coef0)**degree (default=2)
                  
    Notes
    -----
    
    Computational complexity of training (model selection is basically free due to fast leave-one-out):
    m = n_samples, d = n_features, l = n_labels, b = n_bvectors
    
    O(m^3 + lm^2): basic case
    O(dlm + md^2 + d^3): Linear Kernel, d < m
    O(bml + mb^2): Sparse approximation with basis vectors 
     
    Basic information about RLS, and a description of the fast leave-one-out method
    can be found in [1]_. 

    References
    ---------- 
               
    .. [1] Ryan Rifkin, Ross Lippert.
    Notes on Regularized Least Squares
    Technical Report, MIT, 2007.
    """
    
    def __init__(self, X, Y, kernel='LinearKernel', basis_vectors = None, regparams=None, measure=None, **kwargs):
        if regparams == None:
            grid = [2**x for x in range(-15, 16)]
        else:
            grid = regparams
        if measure == None:
            measure = sqerror
        learner = RLS(X, Y, grid[0], kernel, basis_vectors, **kwargs)
        crossvalidator = LOOCV(learner, measure)
        self.cv_performances, self.cv_predictions = grid_search(crossvalidator, grid)
        self.predictor = learner.predictor
            
class KfoldRLS(PredictorInterface):
    
    """Regularized least-squares regression/classification. Wrapper code that selects
    regularization parameter automatically based on K-fold cross-validation.

    Parameters
    ----------
    X: {array-like, sparse matrix}, shape = [n_samples, n_features]
        Data matrix
    Y: {array-like}, shape = [n_samples] or [n_samples, n_labels]
        Training set labels
    folds: list of index lists, shape = [n_folds]
        Each list within the folds list contains the indices of samples in one fold, indices
        must be from range [0,n_samples-1]
    kernel: {'LinearKernel', 'GaussianKernel', 'PolynomialKernel', 'PrecomputedKernel', ...}
        kernel function name, imported dynamically from rlscore.kernel
    basis_vectors: {array-like, sparse matrix}, shape = [n_bvectors, n_features], optional
        basis vectors (typically a randomly chosen subset of the training data)
    regparams: {array-like}, shape = [grid_size] (optional)
        regularization parameter values to be tested, default = [2^-15,...,2^15]
    measure: function(Y, P) (optional)
        a performance measure from rlscore.measure used for model selection,
        default sqerror (squared error)

        
    Other Parameters
    ----------------
    Typical kernel parameters include:
    bias: float, optional
        LinearKernel: the model is w*x + bias*w0, (default=1.0)
    gamma: float, optional
        GaussianKernel: k(xi,xj) = e^(-gamma*<xi-xj,xi-xj>) (default=1.0)
        PolynomialKernel: k(xi,xj) = (gamma * <xi, xj> + coef0)**degree (default=1.0)
    degree: float, optional
        PolynomialKernel: k(xi,xj) = (gamma * <xi, xj> + coef0)**degree (default=1.0)        
    coef0: float, optional
        PolynomialKernel: k(xi,xj) = (gamma * <xi, xj> + coef0)**degree (default=0.)
    degree: int, optional
        PolynomialKernel: k(xi,xj) = (gamma * <xi, xj> + coef0)**degree (default=2)
                  
    Notes
    -----
    
    Computational complexity of training (model selection is basically free due to fast K-fold algorithm):
    m = n_samples, d = n_features, l = n_labels, b = n_bvectors
    
    O(m^3 + lm^2): basic case
    O(dlm + md^2 + d^3): Linear Kernel, d < m
    O(bml + mb^2): Sparse approximation with basis vectors 
     
    Basic information about RLS can be found in [1]_. The K-fold algorithm is based on results published
    in [2]_ and [3]_

    References
    ---------- 
               
    .. [1] Ryan Rifkin, Ross Lippert.
    Notes on Regularized Least Squares
    Technical Report, MIT, 2007.
    
    .. [2] Tapio Pahikkala, Jorma Boberg, and Tapio Salakoski.
    Fast n-Fold Cross-Validation for Regularized Least-Squares.
    Proceedings of the Ninth Scandinavian Conference on Artificial Intelligence,
    83-90, Otamedia Oy, 2006.
        
    .. [3] Tapio Pahikkala, Hanna Suominen, and Jorma Boberg.
    Efficient cross-validation for kernelized least-squares regression with sparse basis expansions.
    Machine Learning, 87(3):381--407, June 2012.   
    """
    
    def __init__(self, X, Y, folds, kernel='LinearKernel', basis_vectors = None, regparams=None, measure=None, save_predictions = False, **kwargs):
        if regparams == None:
            grid = [2**x for x in range(-15, 16)]
        else:
            grid = regparams
        if measure == None:
            self.measure = sqerror
        else:
            self.measure = measure
        learner = RLS(X, Y, grid[0], kernel, basis_vectors, **kwargs)
        crossvalidator = NfoldCV(learner, measure, folds)
        self.cv_performances, self.cv_predictions = grid_search(crossvalidator, grid)
        self.predictor = learner.predictor
        
class LeavePairOutRLS(PredictorInterface):
    
    """Regularized least-squares regression/classification. Wrapper code that selects
    regularization parameter automatically based on ranking accuracy (area under ROC curve
    for binary classification tasks) in K-fold cross-validation.

    Parameters
    ----------
    X: {array-like, sparse matrix}, shape = [n_samples, n_features]
        Data matrix
    Y: {array-like}, shape = [n_samples] or [n_samples, n_labels]
        Training set labels
    kernel: {'LinearKernel', 'GaussianKernel', 'PolynomialKernel', 'PrecomputedKernel', ...}
        kernel function name, imported dynamically from rlscore.kernel
    basis_vectors: {array-like, sparse matrix}, shape = [n_bvectors, n_features], optional
        basis vectors (typically a randomly chosen subset of the training data)
    regparams: {array-like}, shape = [grid_size] (optional)
        regularization parameter values to be tested, default = [2^-15,...,2^15]
    measure: function(Y, P) (optional)
        a performance measure from rlscore.measure used for model selection,
        default sqerror (squared error)

        
    Other Parameters
    ----------------
    Typical kernel parameters include:
    bias: float, optional
        LinearKernel: the model is w*x + bias*w0, (default=1.0)
    gamma: float, optional
        GaussianKernel: k(xi,xj) = e^(-gamma*<xi-xj,xi-xj>) (default=1.0)
        PolynomialKernel: k(xi,xj) = (gamma * <xi, xj> + coef0)**degree (default=1.0)
    degree: float, optional
        PolynomialKernel: k(xi,xj) = (gamma * <xi, xj> + coef0)**degree (default=1.0)        
    coef0: float, optional
        PolynomialKernel: k(xi,xj) = (gamma * <xi, xj> + coef0)**degree (default=0.)
    degree: int, optional
        PolynomialKernel: k(xi,xj) = (gamma * <xi, xj> + coef0)**degree (default=2)
                  
    Notes
    -----
    
    Computational complexity of training and model selection:
    m = n_samples, d = n_features, l = n_labels, b = n_bvectors
    
    O(m^3 + lm^2): basic case
    O(lm^2 + md^2 + d^3): Linear Kernel, d < m
    O(lm^2 + mb^2): Sparse approximation with basis vectors 
     
    Basic information about RLS can be found in [1]_ . The leave-pair-out algorithm
    is an adaptation of the method published in [2]_ . The use of leave-pair-out
    cross-validation for AUC estimation has been analyzed in [3]_.
    
    References
    ----------    
        
    .. [1] Ryan Rifkin, Ross Lippert.
    Notes on Regularized Least Squares
    Technical Report, MIT, 2007.
    
    .. [2] Tapio Pahikkala, Antti Airola, Jorma Boberg, and Tapio Salakoski.
    Exact and efficient leave-pair-out cross-validation for ranking RLS.
    In Proceedings of the 2nd International and Interdisciplinary Conference
    on Adaptive Knowledge Representation and Reasoning (AKRR'08), pages 1-8,
    Espoo, Finland, 2008.
        
    .. [3] Antti Airola, Tapio Pahikkala, Willem Waegeman, Bernard De Baets, and Tapio Salakoski.
    An experimental comparison of cross-validation techniques for estimating the area under the ROC curve.
    Computational Statistics & Data Analysis, 55(4):1828--1844, April 2011. 
    """
    
    def __init__(self, X, Y, kernel='LinearKernel', basis_vectors = None, regparams=None, **kwargs):
        if regparams == None:
            grid = [2**x for x in range(-15, 16)]
        else:
            grid = regparams
        learner = RLS(X, Y, grid[0], kernel, basis_vectors, **kwargs)
        crossvalidator = LPOCV(learner)
        self.cv_performances, self.cv_predictions = grid_search(crossvalidator, grid)
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
            except UndefinedPerformance, e:
                pass
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

    def cv(self, regparam):
        rls = self.rls
        rls.solve(regparam)
        Y = rls.Y
        if Y.shape[1] == 1:
            pairs_start_inds, pairs_end_inds = [], []
            for i in range(Y.shape[0] - 1):
                for j in range(i + 1, Y.shape[0]):
                    if Y[i] > Y[j]:
                        pairs_start_inds.append(i)
                        pairs_end_inds.append(j)
                    elif Y[i] < Y[j]:
                        pairs_start_inds.append(j)
                        pairs_end_inds.append(i)
            if len(pairs_start_inds) == 0:
                raise UndefinedPerformance("All labels are the same")
            pred_start, pred_end = rls.leave_pairs_out(np.array(pairs_start_inds), np.array(pairs_end_inds))
            auc = 0.
            for h in range(len(pred_start)):
                if pred_start[h] > pred_end[h]:
                    auc += 1.
                elif pred_start[h] == pred_end[h]:
                    auc += 0.5
            auc /= len(pairs_start_inds)  
            return auc, (pred_start, pred_end)
        else:
            raise Exception("Model selection with LPOCV is not currently implemented with multi-output data")
