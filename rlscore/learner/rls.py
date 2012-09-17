
from numpy import float64, identity, multiply, mat, zeros, sum
import numpy.linalg as la

from rlscore.learner.abstract_learner import AbstractSvdSupervisedLearner

class RLS(AbstractSvdSupervisedLearner):
    """Regularized least-squares regression/classification.
    
    Implements a training algorithm that is cubic either in the
    number of training examples, or dimensionality of feature
    space (linear kernel).
    
    Computational shortcut for N-fold cross-validation: computeHO
    
    Computational shortcut for leave-one-out: computeLOO
    
    Computational shortcut for parameter selection: solve
    
    There are three ways to supply the training data for the learner.
    
    1. train_features: supply the data matrix directly, by default
    RLS will use the linear kernel.
    
    2. kernel_obj: supply the kernel object that has been initialized
    using the training data.
    
    3. kmatrix: supply user created kernel matrix, in this setting RLS
    is unable to return the model, but you may compute cross-validation
    estimates or access the learned parameters from the variable self.A

    Parameters
    ----------
    train_labels: {array-like}, shape = [n_samples] or [n_samples, n_labels]
        Training set labels
    regparam: float (regparam > 0)
        regularization parameter
    train_features: {array-like, sparse matrix}, shape = [n_samples, n_features], optional
        Data matrix
    kernel_obj: kernel object, optional
        kernel object, initialized with the training set
    kmatrix: : {array-like}, shape = [n_samples, n_samples], optional
        kernel matrix of the training set
    
    References
    ----------
     
    Basic information about RLS, and a description of the fast leave-one-out method
    can be found in [1]_. The efficient N-fold cross-validation algorithm implemented in
    the method computeHO is described in [2]_.
           
    .. [1] Ryan Rifkin, Ross Lippert.
    Notes on Regularized Least Squares
    Technical Report, MIT, 2007.
    
    .. [2] Tapio Pahikkala, Jorma Boberg, and Tapio Salakoski.
    Fast n-Fold Cross-Validation for Regularized Least-Squares.
    Proceedings of the Ninth Scandinavian Conference on Artificial Intelligence,
    83-90, Otamedia Oy, 2006.

    """
   
    def solve(self, regparam):
        """Trains the learning algorithm, using the given regularization parameter.
               
        Parameters
        ----------
        regparam: float (regparam > 0)
            regularization parameter
        """
        
        if not hasattr(self, "multiplyright"):
            #print self.svals.shape
            self.multiplyright = self.svecs.T * self.Y
        
            #Eigenvalues of the kernel matrix
            self.evals = multiply(self.svals, self.svals)
        
        self.newevals = 1. / (self.evals + regparam)
        self.regparam = regparam
        self.A = self.svecs * multiply(self.newevals.T, self.multiplyright)
        #if self.U == None:
        #    pass
            #Dual RLS
            #self.A = self.svecs * multiply(self.newevals.T, self.multiplyright)
        #else:
            #Primal RLS
            #bevals = multiply(self.svals, self.newevals)
            #self.A = self.U.T * multiply(bevals.T, self.multiplyright)
        #    self.A = self.U.T * multiply(self.svals.T, self.svecs.T * self.A)
    
    
    def computeHO(self, indices):
        """Computes hold-out predictions for a trained RLS.
        
        Parameters
        ----------
        indices: list of indices, shape = [n_hsamples]
            list of indices of training examples belonging to the set for which the hold-out predictions are calculated. The list can not be empty.

        Returns
        -------
        F : matrix, shape = [n_hsamples, n_labels]
            holdout predictions
        """
        
        if len(indices) == 0:
            raise Exception('Hold-out predictions can not be computed for an empty hold-out set.')
        
        if len(indices) != len(set(indices)):
            raise Exception('Hold-out can have each index only once.')
        
        bevals = multiply(self.evals, self.newevals)
        A = self.svecs[indices]
        right = self.multiplyright - A.T * self.Y[indices]
        RQY = A * multiply(bevals.T, right)
        B = multiply(bevals.T, A.T)
        if len(indices) <= A.shape[1]:
            I = mat(identity(len(indices)))
            result = la.solve(I - A * B, RQY)
        else:
            I = mat(identity(A.shape[1]))
            result = RQY - A * (la.inv(B * A - I) * (B * RQY))
        return result
    
    
    def computeLOO(self):
        """Computes leave-one-out predictions for a trained RLS.
        
        Returns
        -------
        F : matrix, shape = [n_samples, n_labels]
            leave-one-out predictions
        """
        #LOO = mat(zeros((self.size, self.ysize), dtype=float64))
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
        return LOO
    


