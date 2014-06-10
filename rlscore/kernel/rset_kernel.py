'''
Created on May 22, 2011

@author: aatapa
'''

import numpy.linalg as la
import numpy as np

class RsetKernel(object):
    '''
    This class is for testing reduced set approximation.
    '''
    
    '''
    def __init__(self):
        """Default implementation uses the scipy sparse matrices for internal representation of the data."""
        self.bias = 0.
    '''

    def createKernel(cls, **kwargs):
        kernel = cls()
        kernel.base_kernel = kwargs["base_kernel"]
        kernel.basis_features = kwargs["basis_features"]
        kernel.buildPredictionCache(kwargs['train_features'])
        return kernel
    createKernel = classmethod(createKernel)
    
    
    
    def buildPredictionCache(self, train_X, basis_vectors = None):
        Krr = self.base_kernel.getKM(self.basis_features)
        K_r = self.base_kernel.getKM(train_X)
        invKrr = la.inv(Krr)
        self.predcache = np.dot(K_r, invKrr)
    
    
    def getKM(self, test_X):
        Ktr = self.base_kernel.getKM(test_X)
        return np.dot(Ktr, self.predcache.T)
