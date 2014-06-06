import numpy as np


class AbstractKernel(object):
    """The abstract base class from which all kernel implementations
    should be derived."""

    
    def kernel(self, x, z):
        """Kernel function is evaluated with the given arguments x and z.
        
        This function should be overridden by the subclasses. Overriding this
        method is the easiest way to define a new kernel function, though for
        efficiency reasons it is preferable to override getKM directly.
        
        Parameters
        ----------
        x: {array-like, sparse matrix}, shape = [n_features]
        z: {array-like, sparse matrix}, shape = [n_features]
        
        Returns
        -------
        k(x,z): float
        """
        return 0
    
    
    def getKM(self, test_X):
        """Returns the kernel matrix between the basis vectors and X.
        
        Parameters
        ----------
        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
        
        Returns
        -------
        K : array, shape = [n_samples, n_bvectors]
            kernel matrix
        """
        m = len(train_X)
        n = len(test_X)
        K = np.zeros((m, n), dtype=np.float64)
        for i in range(m):
            for j in range(n):
                x1 = self.train_X[i]
                x2 = test_X[j]
                K[i,j] = self.kernel(x1, x2)
        if self.bias != 0:
            K += self.bias
        return K.T

