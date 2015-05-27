from math import sqrt

import numpy as np
import numpy.linalg as la
from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg
import scipy.sparse as sp

from rlscore.learner.abstract_learner import AbstractSupervisedLearner
from rlscore.learner.abstract_learner import AbstractIterativeLearner
from rlscore import model
from rlscore.utilities import array_tools
from rlscore.measure import measure_utilities
from rlscore.measure import sqmprank

class PPRankRLS(AbstractIterativeLearner):
    """Conjugate gradient RankRLS.
    
    Trains linear RankRLS using the conjugate gradient training algorithm. Suitable for
    large high-dimensional but sparse data.
    
    There are three ways to supply the pairwise preferences for the training set, depending
    on the arguments supplied by the user.
    
    1. train_labels: pairwise preferences constructed between all data point pairs
    
    2. train_labels, train_qids: pairwise preferences constructed between all data
    points belonging to the same query.
    
    3. train_preferences: arbitrary pairwise preferences supplied directly by the user.
    
    In order to make training faster, one can use the early stopping technique by
    supplying a separate validationset to be used for determining, when to terminate
    optimization. In this case, training stops once validation set error has failed to
    decrease for ten consequtive iterations. In this case, the caller should
    provide the parameters validation_features, validation_labels and optionally, validation_qids.
    Currently, this option is not supported when learning directly from pairwise
    preferences. 

    Parameters
    ----------
    train_features: {array-like, sparse matrix}, shape = [n_samples, n_features]
        Data matrix
    regparam: float (regparam > 0)
        regularization parameter
    train_labels: {array-like}, shape = [n_samples] or [n_samples, 1], optional
        Training set labels (alternative to: 'train_preferences')
    train_qids: list of n_queries index lists, optional
        Training set qids,  (can be supplied with 'train_labels')
    train_preferences: {array-like}, shape = [n_preferences, 2], optional
        Pairwise preference indices (alternative to: 'train_labels')
        The array contains pairwise preferences one pair per row, i.e. the data point
        corresponding to the first index is preferred over the data point corresponding
        to the second index.
    validation_features:: {array-like, sparse matrix}, shape = [n_samples, n_features], optional
        Data matrix for validation set, needed if early stopping used
    validation_labels: {array-like}, shape = [n_samples] or [n_samples, 1], optional
        Validation set labels, needed if early stopping used
    validation_qids: list of n_queries index lists, optional, optional
        Validation set qids, may be used with early stopping
 
       
    References
    ----------
    
    RankRLS algorithm is described in [1]_, using the conjugate gradient optimization
    together with early stopping was considered in detail in [2]_. 
    
    .. [1] Tapio Pahikkala, Evgeni Tsivtsivadze, Antti Airola, Jouni Jarvinen, and Jorma Boberg.
    An efficient algorithm for learning to rank from preference graphs.
    Machine Learning, 75(1):129-165, 2009.
    
    .. [2] Antti Airola, Tapio Pahikkala, and Tapio Salakoski.
    Large Scale Training Methods for Linear RankRLS
    ECML/PKDD-10 Workshop on Preference Learning, 2010.
    """

    def __init__(self, **kwargs):
        super(PPRankRLS, self).__init__(**kwargs)
        if kwargs.has_key("regparam"):
            self.regparam = float(kwargs["regparam"])
        else:
            self.regparam = 0.
        self.pairs = kwargs['train_preferences']
        self.learn_from_labels = False
        X = kwargs['train_features']
        self.X = csc_matrix(X)
        self.bias = 0.
        self.results = {}
    
    
    def solve(self, regparam):
        """Trains the learning algorithm, using the given regularization parameter.
        
        This implementation simply changes the regparam, and then calls the train method.
        
        Parameters
        ----------
        regparam: float (regparam > 0)
            regularization parameter
        """
        self.regparam = regparam
        self.train()
    
    
    def train(self):
        """Trains the learning algorithm.
        
        After the learner is trained, one can call the method getModel
        to get the trained model
        """
        regparam = self.regparam
        X = self.X.tocsc()
        X_csr = X.tocsr()
        vals = np.concatenate([np.ones((self.pairs.shape[0]), dtype=np.float64), -np.ones((self.pairs.shape[0]), dtype = np.float64)])
        row = np.concatenate([np.arange(self.pairs.shape[0]), np.arange(self.pairs.shape[0])])
        col = np.concatenate([self.pairs[:,0], self.pairs[:,1]])
        coo = coo_matrix((vals, (row, col)), shape = (self.pairs.shape[0], X.shape[0]))
        #pairs_csr = coo.tocsr()
        #pairs_csc = coo.tocsc()
        
        In = np.mat(np.identity(X.shape[1]))
        
        L = (coo.T * coo).todense()
        C = X.T * L * X + regparam*In
        
        W = np.squeeze(np.array(C.I * (X.T * (coo.T * np.mat(np.ones((self.pairs.shape[0], 1)))))))
        
        
        '''
        def mv(v):
            vmat = np.mat(v).T
            ret = np.array(X_csr * (pairs_csc.T * (pairs_csr * (X.T * vmat))))+regparam*vmat
            return ret
        G = LinearOperator((X.shape[0], X.shape[0]), matvec=mv, dtype=np.float64)
        M = np.mat(np.ones((self.pairs.shape[0], 1)))
        if not self.callbackfun == None:
            def cb(v):
                self.A = np.mat(v).T
                self.b = np.mat(np.zeros((1,1)))
                self.callback()
        else:
            cb = None
        XLY = X_csr * (pairs_csc.T * M)
        '''
        self.A = W
        self.b = np.mat(np.zeros((1, 1)))
        self.results['model'] = self.getModel()
        
    
    
    def getModel(self):
        """Returns the trained model, call this only after training.
        
        Returns
        -------
        model : LinearModel
            prediction function
        """
        return model.LinearModel(self.A, self.b)
    