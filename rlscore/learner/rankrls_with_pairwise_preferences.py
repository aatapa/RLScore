import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix


from rlscore.utilities import creators
from rlscore.utilities import decomposition
import numpy as np

class PPRankRLS(object):
    """
    """


    def __init__(self, X, pairs_start_inds, pairs_end_inds, regparam = 1.0, kernel='LinearKernel', basis_vectors = None, **kwargs):
        kwargs['kernel'] =  kernel
        kwargs['X'] = X
        if basis_vectors != None:
            kwargs["basis_vectors"] = basis_vectors
        self.regparam = regparam
        self.pairs = np.vstack([pairs_start_inds, pairs_end_inds]).T
        self.svdad = creators.createSVDAdapter(**kwargs)
        #self.Y = array_tools.as_labelmatrix(kwargs["Y"])
        #if kwargs.has_key("regparam"):
        #    self.regparam = float(kwargs["regparam"])
        #else:
        #    self.regparam = 1.
        self.svals = self.svdad.svals
        self.svecs = self.svdad.rsvecs
        self.results = {}
        self.X = csc_matrix(X)
        self.bias = 0.
        self.results = {}
        self.train()
    
    
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
        size = self.svecs.shape[0]
        
        if not hasattr(self, "multipleright"):
            vals = np.concatenate([np.ones((self.pairs.shape[0]), dtype=np.float64), -np.ones((self.pairs.shape[0]), dtype = np.float64)])
            row = np.concatenate([np.arange(self.pairs.shape[0]), np.arange(self.pairs.shape[0])])
            col = np.concatenate([self.pairs[:, 0], self.pairs[:, 1]])
            coo = coo_matrix((vals, (row, col)), shape = (self.pairs.shape[0], size))
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
        #self.results['model'] = self.getModel()
        self.predictor = self.svdad.createModel(self)
        
    def predict(self, X):
        return self.predictor.predict(X)
        
    
    