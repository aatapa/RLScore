import numpy.linalg as la
import numpy as np

class RsetKernel(object):
    '''
    This class is for testing reduced set approximation.
    '''
    
    def __init__(self, base_kernel, X, basis_features):
        """Default implementation uses the scipy sparse matrices for internal representation of the data."""
        self.base_kernel = base_kernel
        Krr = self.base_kernel.getKM(basis_features)
        K_r = self.base_kernel.getKM(X)
        invKrr = la.inv(Krr)
        self.predcache = np.dot(K_r, invKrr)
        self.train_X = X

    
    def getKM(self, test_X):
        Ktr = self.base_kernel.getKM(test_X)
        return np.dot(Ktr, self.predcache.T)
    
    