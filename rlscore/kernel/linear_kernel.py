from scipy import sparse as sp

from rlscore.utilities import array_tools

class LinearKernel(object):
    """Linear kernel.
    
    k(xi,xj) = <xi , xj> + bias

    Parameters
    ----------
    train_features: {array-like, sparse matrix}, shape = [n_samples, n_features]
        Data matrix
    bias : float, optional (default 0.)
        Constant added to each kernel evaluation
    basis_vectors : array of integers, shape = [n_bvectors] or None, optional (default None)
        Indices for the subset of rows of X to be used as basis vectors. If set to None,
        by default basis_vectors = range(n_samples).
    """

    def __init__(self, train_features, bias=0.0, basis_vectors=None, **kwargs):
        if basis_vectors != None:
            self.train_X = basis_vectors
        else:
            self.train_X = train_features
        self.bias = bias

    def createKernel(cls, **kwargs):
        """Initializes a kernel object from the arguments."""
        #new_kwargs = {}
        #if kwargs.has_key('basis_vectors'):
        #    new_kwargs['basis_vectors'] = kwargs['basis_vectors']
        #new_kwargs["train_features"] = kwargs["train_features"]
        #if "bias" in kwargs:
        #    new_kwargs["bias"] = float(kwargs["bias"])
        #kernel = cls(**new_kwargs)
        kernel = cls(**kwargs)
        return kernel
    createKernel = classmethod(createKernel)  
    
    def getKM(self, X):
        """Returns the kernel matrix between the basis vectors and X.
        
        Parameters
        ----------
        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
        
        Returns
        -------
        K : array, shape = [n_samples, n_bvectors]
            kernel matrix
        """
        test_X = X
        if sp.issparse(test_X):
            test_X = array_tools.spmat_resize(test_X, self.train_X.shape[1])
        else:
            test_X = array_tools.as_dense_matrix(test_X)
        train_X = self.train_X
        K = train_X * test_X.T
        K = array_tools.as_array(K)
        if self.bias != 0:
            K += self.bias
        return K.T

