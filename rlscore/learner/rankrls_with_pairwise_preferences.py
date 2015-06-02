from math import sqrt

import numpy as np
import numpy.linalg as la
from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg
import scipy.sparse as sp

from rlscore.learner.abstract_learner import AbstractSupervisedLearner
from rlscore.learner.abstract_learner import AbstractSvdLearner
from rlscore import model
from rlscore.utilities import array_tools
from rlscore.measure import measure_utilities
from rlscore.measure import sqmprank
from rlscore.utilities import creators
from rlscore.utilities import decomposition

class PPRankRLS(AbstractSvdLearner):
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
        self.svdad = creators.createSVDAdapter(**kwargs)
        #self.Y = array_tools.as_labelmatrix(kwargs["train_labels"])
        #if kwargs.has_key("regparam"):
        #    self.regparam = float(kwargs["regparam"])
        #else:
        #    self.regparam = 1.
        self.svals = self.svdad.svals
        self.svecs = self.svdad.rsvecs
        self.results = {}
        X = kwargs['train_features']
        self.X = csc_matrix(X)
        self.bias = 0.
        self.results = {}
    
    
    def train(self):
        """Trains the prediction function.
        
        After the learner is trained, one can call the method getModel
        to get the trained model
        """
        regparam = self.regparam
        self.solve(regparam)
    
    
    def solve(self, regparam):
        """Trains the prediction function, using the given regularization parameter.
        
        This implementation simply changes the regparam, and then calls the train method.
        
        Parameters
        ----------
        regparam: float (regparam > 0)
            regularization parameter
        """
        
        if not hasattr(self, "multipleright"):
            vals = np.concatenate([np.ones((self.pairs.shape[0]), dtype=np.float64), -np.ones((self.pairs.shape[0]), dtype = np.float64)])
            row = np.concatenate([np.arange(self.pairs.shape[0]), np.arange(self.pairs.shape[0])])
            col = np.concatenate([self.pairs[:, 0], self.pairs[:, 1]])
            coo = coo_matrix((vals, (row, col)), shape = (self.pairs.shape[0], self.size))
            self.L = (coo.T * coo)#.todense()
            
            #Eigenvalues of the kernel matrix
            evals = np.multiply(self.svals, self.svals)
            
            #Temporary variables
            ssvecs = np.multiply(self.svecs, self.svals)
            
            #These are cached for later use in solve and computeHO functions
            ssvecsTLssvecs = ssvecs.T * self.L * ssvecs
            LRsvals, LRevecs = decomposition.decomposeKernelMatrix(ssvecsTLssvecs)
            LRevals = np.multiply(LRsvals, LRsvals)
            LY = coo.T * np.mat(np.ones((self.pairs.shape[0], 1)))
            self.multipleright = LRevecs.T * (ssvecs.T * LY)
            self.multipleleft = ssvecs * LRevecs
            self.LRevals = LRevals
            self.LRevecs = LRevecs
        
        
        self.regparam = regparam
        
        #Compute the eigenvalues determined by the given regularization parameter
        self.neweigvals = 1. / (self.LRevals + regparam)
        self.A = self.svecs * np.multiply(1. / self.svals.T, (self.LRevecs * np.multiply(self.neweigvals.T, self.multipleright)))
        self.results['model'] = self.getModel()
        
    
    