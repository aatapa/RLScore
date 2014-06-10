
from scipy import sparse as sp
import numpy as np

from rlscore.utilities import array_tools


class DualModel(object):
    """Represents a dual model for making predictions.
    
    New predictions are made by computing K*A, where K is the
    kernel matrix between test and training examples, and A contains
    the dual coefficients.

    Parameters
    ----------
    A: sparse matrix, shape = [n_samples, n_tasks]
        dual coefficients
    kernel : kernel object
        kernel object, initialized with the basis vectors and kernel parameters
    rpool: resource pool
    """
    
    def __init__(self, A, kernel):
        self.kernel = kernel
        #newbasis = list(set(nonz[0].tolist()))
        self.A = A
        #if len(newbasis) != A.shape[0]:
        #    self.A = A.todense()[newbasis]
        #    newpool = rpool.copy()
        #    newpool['basis_vectors'] = newbasis
        #    self.kernel = kernel.__class__.createKernel(**newpool)
        self.A = array_tools.as_array(self.A)

    
    def predictFromPool(self, rpool):
        """Makes real-valued predictions for new examples"""
        K = self.kernel.getKM(rpool['prediction_features'])
        P = np.dot(K, self.A)
        P = array_tools.as_array(P)
        return P
    
    
    def predict(self, X):
        """Computes predictions for test examples.

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            test data matrix
        
        Returns
        ----------
        P: array, shape = [n_samples, n_tasks]
            predictions
        """
        K = self.kernel.getKM(X)
        P = np.dot(K, self.A)
        P = array_tools.as_array(P)
        return P


class LinearModel(object):
    """Represents a linear model for making predictions.
    
    New predictions are made by computing X*W+b.

    Parameters
    ----------
    W: sparse matrix, shape = [n_features, n_tasks]
        primal coefficients
    b : array-line, shape = [n_tasks]
        vector of bias terms
    """
    
    def __init__(self, W, b):
        """Initializes a primal model
        @param W: coefficients of the linear model, one column per task
        @type W: numpy matrix
        @param b: bias of the model, one column per task
        @type b: numpy matrix
        """
        self.W = array_tools.as_dense_matrix(W)
        self.b = b
    
    
    def predictFromPool(self, rpool):
        """Makes real-valued predictions for new examples"""
        X = rpool['prediction_features']
        return self.predict(X)
    
    
    def predict(self, X):
        """Computes predictions for test examples.

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            test data matrix
        
        Returns
        ----------
        P: array, shape = [n_samples, n_tasks]
            predictions
        """
        W = self.W
        if X.shape[1] > W.shape[0]:
            #print 'Warning: the number of features ('+str(X.shape[0])+') in the data point for which the prediction is to be made is larger than the size ('+str(self.W.shape[0])+') of the predictor. Slicing the feature vector accordingly.'
            X = X[:,range(W.shape[0])]
        if X.shape[1] < W.shape[0]:
            #print 'Warning: the number of features ('+str(X.shape[0])+') in the data point for which the prediction is to be made is smaller than the size ('+str(self.W.shape[0])+') of the predictor. Slicing the predictor accordingly.'
            W = W[range(X.shape[1])]
        if sp.issparse(X):
            P = X * W
        else:
            P = np.dot(X, W)
        P = P + self.b
        return P

