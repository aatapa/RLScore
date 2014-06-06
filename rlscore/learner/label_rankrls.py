import math

from numpy import arange, float64, identity, multiply, mat, ones, zeros
import numpy.linalg as la
import scipy.sparse

from rlscore.utilities import decomposition
from rlscore.learner.abstract_learner import AbstractSvdSupervisedLearner
from rlscore.utilities import array_tools
from rlscore.utilities import creators

class LabelRankRLS(AbstractSvdSupervisedLearner):
    """RankRLS algorithm for learning to rank
    
    Implements the learning algorithm for learning from query-structured
    data. For other settings, see AllPairsRankRLS. Uses a training algorithm
    that is cubic either in the number of training examples, or dimensionality
    of feature space (linear kernel).
    
    Computational shortcut for leave-query-out cross-validation: computeHO
    
    Computational shortcut for parameter selection: solve
    
    There are three ways to supply the training data for the learner.
    
    1. train_features: supply the data matrix directly, by default
    RLS will use the linear kernel.
    
    2. kernel_obj: supply the kernel object that has been initialized
    using the training data.
    
    3. kernel_matrix: supply user created kernel matrix, in this setting RLS
    is unable to return the model, but you may compute cross-validation
    estimates or access the learned parameters from the variable self.A

    Parameters
    ----------
    train_labels: {array-like}, shape = [n_samples] or [n_samples, n_labels]
        Training set labels
    train_qids: list of n_queries index lists
        Training set qids
    regparam: float (regparam > 0)
        regularization parameter
    train_features: {array-like, sparse matrix}, shape = [n_samples, n_features], optional
        Data matrix
    kernel_obj: kernel object, optional
        kernel object, initialized with the training set
    kernel_matrix: : {array-like}, shape = [n_samples, n_samples], optional
        kernel matrix of the training set
        
    References
    ----------
    RankRLS algorithm and the leave-query-out cross-validation method implemented in
    the method 'computeHO' are described in [1]_.

    .. [1] Tapio Pahikkala, Evgeni Tsivtsivadze, Antti Airola, Jouni Jarvinen, and Jorma Boberg.
    An efficient algorithm for learning to rank from preference graphs.
    Machine Learning, 75(1):129-165, 2009.
    """

    def __init__(self, svdad, train_labels, train_qids, regparam=1.0):
        self.svdad = svdad
        self.Y = array_tools.as_labelmatrix(train_labels)
        self.size = self.Y.shape[0]
        self.regparam = regparam
        self.svals = svdad.svals
        self.svecs = svdad.rsvecs
        self.setQids(train_qids)
        self.results = {}
    
    
    def createLearner(cls, **kwargs):
        new_kwargs = {}
        new_kwargs["svdad"] = creators.createSVDAdapter(**kwargs)
        new_kwargs["train_labels"] = kwargs["train_labels"]
        new_kwargs["train_qids"] = kwargs["train_qids"]
        if kwargs.has_key("regparam"):
            new_kwargs['regparam'] = float(kwargs["regparam"])
        learner = cls(**new_kwargs)
        return learner
    createLearner = classmethod(createLearner)
    
    
    def setQids(self, qids):
        """Sets the qid parameters of the training examples. The list must have as many qids as there are training examples.
        
        @param qids: A list of qid parameters.
        @type qids: List of integers."""
        
        self.qidlist = [-1 for i in range(self.size)]
        for i in range(len(qids)):
            for j in qids[i]:
                if j >= self.size:
                    raise Exception("Index %d in query out of training set index bounds" %j)
                elif j < 0:
                    raise Exception("Negative index %d in query, query indices must be non-negative" %j)
                else:
                    self.qidlist[j] = i
        if -1 in self.qidlist:
            raise Exception("Not all training examples were assigned a query")
        
        self.qidmap = {}
        for i in range(len(self.qidlist)):
            qid = self.qidlist[i]
            if self.qidmap.has_key(qid):
                sameqids = self.qidmap[qid]
                sameqids.append(i)
            else:
                self.qidmap[qid] = [i]  
    
    
    def solve(self, regparam=1.0):
        """Trains the learning algorithm, using the given regularization parameter.
               
        Parameters
        ----------
        regparam: float (regparam > 0)
            regularization parameter
        """
        if not hasattr(self, "D"):
            qidlist = self.qidlist
            objcount = max(qidlist) + 1
            
            labelcounts = mat(zeros((1, objcount)))
            Pvals = ones(self.size)
            for i in range(self.size):
                qid = qidlist[i]
                labelcounts[0, qid] = labelcounts[0, qid] + 1
            D = mat(ones((1, self.size), dtype=float64))
            
            #The centering matrix way (HO computations should be modified accordingly too)
            #for i in range(self.size):
            #    qid = qidlist[i]
            #    Pvals[i] = 1. / math.sqrt(labelcounts[0, qid])
            
            #The old Laplacian matrix way
            for i in range(self.size):
                qid = qidlist[i]
                D[0, i] = labelcounts[0, qid]
            
            P = scipy.sparse.coo_matrix((Pvals, (arange(0, self.size), qidlist)), shape=(self.size,objcount))
            P_csc = P.tocsc()
            P_csr = P.tocsr()
            
            
            #Eigenvalues of the kernel matrix
            evals = multiply(self.svals, self.svals)
            
            #Temporary variables
            ssvecs = multiply(self.svecs, self.svals)
            
            #These are cached for later use in solve and computeHO functions
            ssvecsTLssvecs = (multiply(ssvecs.T, D) - (ssvecs.T * P_csc) * P_csr.T) * ssvecs
            LRsvals, LRevecs = decomposition.decomposeKernelMatrix(ssvecsTLssvecs)
            LRevals = multiply(LRsvals, LRsvals)
            LY = multiply(D.T, self.Y) - P_csr * (P_csc.T * self.Y)
            self.multipleright = LRevecs.T * (ssvecs.T * LY)
            self.multipleleft = ssvecs * LRevecs
            self.LRevals = LRevals
            self.LRevecs = LRevecs
            self.D = D
        
        
        self.regparam = regparam
        
        #Compute the eigenvalues determined by the given regularization parameter
        self.neweigvals = 1. / (self.LRevals + regparam)
        self.A = self.svecs * multiply(1. / self.svals.T, (self.LRevecs * multiply(self.neweigvals.T, self.multipleright)))
        #if self.U == None:
            #Dual RLS
        #    pass
            #self.A = self.svecs * multiply(1. / self.svals.T, (self.LRevecs * multiply(self.neweigvals.T, self.multipleright)))
        #else:
            #Primal RLS
            #self.A = self.U.T * (self.LRevecs * multiply(self.neweigvals.T, self.multipleright))
            #self.A = self.U.T * multiply(self.svals.T,  self.svecs.T * self.A)
        self.results['model'] = self.getModel()
    
    
    def computeHO(self, indices):
        """Computes hold-out predictions for a trained RLS.
        
        Parameters
        ----------
        indices: list of indices, shape = [n_hsamples]
            list of indices of training examples belonging to the set for which the
            hold-out predictions are calculated. Should correspond to one query.

        Returns
        -------
        F : matrix, shape = [n_hsamples, n_labels]
            holdout query predictions
        """
        
        if len(indices) == 0:
            raise Exception('Hold-out predictions can not be computed for an empty hold-out set.')
        
        if len(indices) != len(set(indices)):
            raise Exception('Hold-out can have each index only once.')
        
        hoqid = self.qidlist[indices[0]]
        for ind in indices:
            if not hoqid == self.qidlist[ind]:
                raise Exception('All examples in the hold-out set must have the same qid.')
        
        if not len(self.qidmap[hoqid]) == len(indices):
            raise Exception('All examples in the whole training set having the same qid as the examples in the hold-out set must belong to the hold out set.')
        
        indlen = len(indices)
        Qleft = self.multipleleft[indices]
        Qho = Qleft * multiply(self.neweigvals.T, Qleft.T)
        Pho = mat(ones((len(indices),1)))
        Yho = self.Y[indices]
        Dho = self.D[:, indices]
        LhoYho = multiply(Dho.T, Yho) - Pho * (Pho.T * Yho)
        RQY = Qleft * multiply(self.neweigvals.T, self.multipleright) - Qho * LhoYho
        RQRTLho = multiply(Qho, Dho) - (Qho * Pho) * Pho.T
        I = mat(identity(indlen))
        return la.inv(I - RQRTLho) * RQY
        #return RQY - RQRTLho * la.inv(-I + RQRTLho) * RQY


