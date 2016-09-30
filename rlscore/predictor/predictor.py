#
# The MIT License (MIT)
#
# This file is part of RLScore 
#
# Copyright (c) 2016 Tapio Pahikkala, Antti Airola
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

from scipy import sparse as sp
import numpy as np

from rlscore.utilities import array_tools

class PredictorInterface(object):
    """Predictor interface
    
    Attributes
    ----------
    predictor : predictor object
        predicts outputs for new instance
    """
    

    
    def predict(self, X):
        """Predicts outputs for new inputs
    
        Parameters
         ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
           input data matrix
            
        Returns
        -------
        P : array, shape = [n_samples, n_tasks]
            predictions
        """
        return self.predictor.predict(X)

class KernelPredictor(object):
    """Represents a dual model for making predictions.
    
    New predictions are made by computing K*A, where K is the
    kernel matrix between test and training examples, and A contains
    the dual coefficients.

    Parameters
    ----------
    A : array-like, shape = [n_samples] or [n_samples, n_labels]
        dual coefficients
    kernel : kernel object
        kernel object, initialized with the basis vectors and kernel parameters
        
    Attributes
    ----------
    A : array-like, shape = [n_samples] or [n_samples, n_labels]
        dual coefficients
    kernel : kernel object
        kernel object, initialized with the basis vectors and kernel parameters
    """
    
    def __init__(self, A, kernel):
        self.kernel = kernel
        self.dim = kernel.train_X.shape[1]
        self.A = A
        self.A = np.squeeze(array_tools.as_array(self.A))
    
    
    def predict(self, X):
        """Computes predictions for test examples.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            test data matrix
        
        Returns
        ----------
        P : array, shape = [n_samples] or [n_samples, n_labels]
            predictions
        """
        if len(X.shape) == 1:
            #One dimensional data
            if self.dim == 1:
                X = X.reshape(X.shape[0], 1)
            else:
                X = X.reshape(1, X.shape[0])
        K = self.kernel.getKM(X)
        if len(X.shape) < 2: #Cheap hack!
            K = np.squeeze(K)
        P = np.dot(K, self.A)
        P = np.squeeze(P)
        return P


class LinearPredictor(object):
    """Represents a linear model for making predictions.
    
    New predictions are made by computing X*W+b.

    Parameters
    ----------
    W : array-like, shape = [n_features] or [n_features, n_labels]
        primal coefficients
    b : float or array-like with shape = [n_labels]
        bias term(s)

    Attributes
    ----------
    W : array-like, shape = [n_features] or [n_features, n_labels]
        primal coefficients
    b : float or array-like with shape = [n_labels]
        bias term(s)
    """
    
    def __init__(self, W, b = 0.):
        self.W = np.squeeze(array_tools.as_array(W))
        if self.W.ndim == 0:
            self.W = self.W.reshape(1)
        #special case: 1-dimensional multi-task predictor
        if W.shape[0] == 1 and W.shape[1] > 0:
            self.W = self.W.reshape(W.shape[0], W.shape[1])
        self.b = np.squeeze(np.array(b))
    
    
    def predict(self, X):
        """Computes predictions for test examples.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            test data matrix
        
        Returns
        ----------
        P : array, shape = [n_samples, n_labels]
            predictions
        """
        W = self.W
        if len(X.shape) == 1:
            #One dimensional data
            if len(W) == 1:
                X = X.reshape(X.shape[0], 1)
            else:
                X = X.reshape(1, X.shape[0])
        assert len(X.shape) < 3
        if sp.issparse(X):
            P = X * W
        elif isinstance(X, np.matrix):
            P = np.dot(np.array(X), W)
        else:
            P = np.dot(X, W)
        P = P + self.b
        #P = array_tools.as_array(P)
        P = np.squeeze(P)
        return P



