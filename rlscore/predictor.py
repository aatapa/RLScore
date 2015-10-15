
from scipy import sparse as sp
import numpy as np

from rlscore.utilities import array_tools

class PredictorInterface(object):
    """Predicts outputs for new inputs.

    Parameters
     ----------
    X: {array-like, sparse matrix}, shape = [n_samples, n_features]
       input data matrix
        
    Returns
     ----------
    P: array, shape = [n_samples, n_tasks]
        predictions
    """
    
    def predict(self, X):
        return self.predictor.predict(X)

class KernelPredictor(object):
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
        self.A = np.squeeze(array_tools.as_array(self.A))
    
    
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
        if len(X.shape) < 2: #Cheap hack!
            K = np.squeeze(K)
        P = np.dot(K, self.A)
        return P


class LinearPredictor(object):
    """Represents a linear model for making predictions.
    
    New predictions are made by computing X*W+b.

    Parameters
    ----------
    W: sparse matrix, shape = [n_features, n_tasks]
        primal coefficients
    b : array-line, shape = [n_tasks]
        vector of bias terms
    """
    
    def __init__(self, W, b = 0.):
        """Initializes a primal model
        @param W: coefficients of the linear model, one column per task
        @type W: numpy matrix
        @param b: bias of the model, one column per task
        @type b: numpy matrix
        """
        #self.W = array_tools.as_dense_matrix(W)
        #print type(W), W.shape
        self.W = np.squeeze(array_tools.as_array(W))
        if self.W.ndim == 0:
            self.W = self.W.reshape(1)
        #print type(self.W), self.W.shape
        self.b = np.squeeze(np.array(b))
    
    
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
        assert len(X.shape) < 3
        xfcount = X.shape[len(X.shape) - 1]
        if xfcount > W.shape[0]:
            #print 'Warning: the number of features ('+str(X.shape[0])+') in the data point for which the prediction is to be made is larger than the size ('+str(self.W.shape[0])+') of the predictor. Slicing the feature vector accordingly.'
            X = X[:, range(W.shape[0])]
        if xfcount < W.shape[0]:
            #print 'Warning: the number of features ('+str(X.shape[0])+') in the data point for which the prediction is to be made is smaller than the size ('+str(self.W.shape[0])+') of the predictor. Slicing the predictor accordingly.'
            W = W[range(xfcount)]
        if sp.issparse(X):
            P = X * W
        elif isinstance(X, np.matrix):
            P = np.dot(np.array(X), W)
        else:
            P = np.dot(X, W)
        P = P + self.b
        #P = array_tools.as_array(P)
        return P



