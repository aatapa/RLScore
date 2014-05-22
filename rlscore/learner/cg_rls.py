from math import sqrt

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg
from scipy import sparse
from numpy import ones

from rlscore.learner.abstract_learner import AbstractSupervisedLearner
from rlscore.learner.abstract_learner import AbstractIterativeLearner
from rlscore import data_sources
from rlscore.utilities import array_tools
from rlscore import model
from rlscore.measure import sqerror

class CGRLS(AbstractSupervisedLearner, AbstractIterativeLearner):
    """Conjugate gradient RLS.
    
    Trains linear RLS using the conjugate gradient training algorithm. Suitable for
    large high-dimensional but sparse data.
    
    In order to make training faster, one can use the early stopping technique by
    supplying a separate validationset to be used for determining, when to terminate
    optimization. In this case, training stops once validation set error has failed to
    decrease for ten consecutive iterations. In this case, the caller should
    provide the parameters validation_features and validation_labels. 

    Parameters
    ----------
    train_features: {array-like, sparse matrix}, shape = [n_samples, n_features]
        Data matrix
    regparam: float (regparam > 0)
        regularization parameter
    train_labels: {array-like}, shape = [n_samples] or [n_samples, 1]
        Training set labels
    validation_features:: {array-like, sparse matrix}, shape = [n_samples, n_features], optional
        Data matrix for validation set, needed if early stopping used
    validation_labels: {array-like}, shape = [n_samples] or [n_samples, 1], optional
        Validation set labels, needed if early stopping used
    bias: float, optional
        value of constant feature added to each data point (default 0)
        
    References
    ----------
    
    For an overview of regularized least-squares, and the conjugate gradient based optimization
    scheme see  [1]_.
    
    .. [1] Ryan Rifkin
    Everything old is new again : a fresh look at historical approaches in machine learning
    PhD Thesis, Massachusetts Institute of Technology, 2002
    """

    def __init__(self, train_features, train_labels, validation_features=None, validation_labels=None, regparam=1.0, bias=1.0):
        X = train_features
        self.Y = array_tools.as_labelmatrix(train_labels)
        self.X = csc_matrix(X.T)
        self.bias = bias
        self.regparam = regparam
        if self.bias != 0.:
            bias_slice = sqrt(self.bias)*np.mat(ones((1,self.X.shape[1]),dtype=np.float64))
            self.X = sparse.vstack([self.X,bias_slice]).tocsc()
        else:
            self.bias = 0.
        self.X_csr = self.X.tocsr()
        if validation_features != None and validation_labels != None:
            self.callbackfun = EarlyStopCB(validation_features, validation_labels)
        else:
            self.callbackfun = None
        self.results = {}

    def createLearner(cls, **kwargs):
        new_kwargs = {}
        new_kwargs["train_features"] = kwargs["train_features"]
        new_kwargs["train_labels"] = kwargs["train_labels"]
        if kwargs.has_key("regparam"):
            new_kwargs['regparam'] = float(kwargs["regparam"])
        if kwargs.has_key("bias"):
            new_kwargs['bias'] = float(kwargs["bias"])
        if kwargs.has_key(data_sources.VALIDATION_FEATURES) and kwargs.has_key(data_sources.VALIDATION_LABELS):
            new_kwargs[data_sources.VALIDATION_FEATURES] = kwargs[data_sources.VALIDATION_FEATURES]
            new_kwargs[data_sources.VALIDATION_LABELS] = kwargs[data_sources.VALIDATION_LABELS]
        learner = cls(**new_kwargs)
        return learner
    createLearner = classmethod(createLearner)

    def loadResources(self):
        AbstractIterativeLearner.loadResources(self)
        AbstractSupervisedLearner.loadResources(self)
        X = self.resource_pool[data_sources.TRAIN_FEATURES]
        self.X = csc_matrix(X.T)
        if self.resource_pool.has_key('bias'):
            self.bias = float(self.resource_pool['bias'])
            if self.bias != 0.:
                bias_slice = sqrt(self.bias)*np.mat(ones((1,self.X.shape[1]),dtype=np.float64))
                self.X = sparse.vstack([self.X,bias_slice]).tocsc()
        else:
            self.bias = 0.
        self.X_csr = self.X.tocsr()
        if (data_sources.VALIDATION_FEATURES in self.resource_pool) and (data_sources.VALIDATION_LABELS in self.resource_pool):
            validation_X = self.resource_pool[data_sources.VALIDATION_FEATURES]
            validation_Y = self.resource_pool[data_sources.VALIDATION_LABELS]
            self.callbackfun = EarlyStopCB(validation_X, validation_Y)
    
    
    def solve(self, regparam):
        """Trains the learning algorithm, using the given regularization parameter.

        This implementation simply changes the regparam, and then calls the train method.
               
        Parameters
        ----------
        regparam: float (regparam > 0)
            regularization parameter
        """
        #self.resource_pool[data_sources.TIKHONOV_REGULARIZATION_PARAMETER] = regparam
        self.regparam = regparam
        self.train()   
    
    
    def train(self):
        """Trains the learning algorithm.
        
        After the learner is trained, one can call the method getModel
        to get the trained model
        """
        regparam = self.regparam
        Y = self.Y
        X = self.X
        X_csr = self.X_csr
        def mv(v):
            return X.T*(X_csr*v)+regparam*v
        G = LinearOperator((X.shape[1],X.shape[1]), matvec=mv, dtype=np.float64)
        self.AA = []
        if not self.callbackfun == None:
            def cb(v):
                self.A = np.mat(v).T
                self.callback()
        else:
            cb = None
        try:
            self.A = np.mat(cg(G, Y, callback=cb)[0]).T
        except Finished, e:
            pass
        self.finished()
        self.A = X_csr*self.A
        if self.bias == 0.:
            self.b = np.mat(np.zeros((1,1)))
        else:
            self.b = sqrt(self.bias)*self.A[-1]
            self.A = self.A[:-1]
        self.results[data_sources.MODEL] = self.getModel()

    def getModel(self):
        """Returns the trained model, call this only after training.
        
        Returns
        -------
        model : LinearModel
            prediction function
        """
        return model.LinearModel(self.A, self.b)
    

class EarlyStopCB(object):
    
    def __init__(self, X_valid, Y_valid, measure=sqerror, maxiter=10):
        self.X_valid = array_tools.as_matrix(X_valid)
        self.Y_valid = array_tools.as_labelmatrix(Y_valid)
        self.measure = measure
        self.bestperf = None
        self.bestA = None
        self.iter = 0
        self.last_update = 0
        self.maxiter = maxiter
    
    def callback(self, learner):
        A = learner.A
        b = learner.bias
        A = learner.X_csr*A
        if b == 0:
            b = np.mat(np.zeros((1,1)))
        else:
            b = sqrt(b)*A[-1]
            A = A[:-1]
        m = model.LinearModel(A,b)
        P = m.predict(self.X_valid)
        perf = self.measure(self.Y_valid,P)
        if self.bestperf == None or (self.measure.iserror == (perf < self.bestperf)):
            self.bestperf = perf
            self.bestA = learner.A
            self.last_update = 0
        else:
            self.iter += 1
            self.last_update += 1
        #print self.iter,  self.last_update, perf
        if self.last_update == self.maxiter:
            learner.A = np.mat(self.bestA)
            raise Finished("Done")

        
    def finished(self, learner):
        pass
        

class Finished(Exception):
    """Used to indicate that the optimization is finished and should
    be terminated."""

    def __init__(self, value):
        """Initialization
        
        @param value: the error message
        @type value: string"""
        self.value = value

    def __str__(self):
        return repr(self.value)  

    