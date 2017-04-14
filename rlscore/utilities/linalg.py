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

import numpy as np
import numpy.linalg as la
from numpy.linalg.linalg import LinAlgError

SMALLEST_EVAL = 0.0000000001

def svd_economy_sized(X):
    """Returns the reduced singular value decomposition of the data matrix X so that only the singular vectors corresponding to the nonzero singular values are returned.
    
    @param X: data matrix whose rows and columns correspond to the data and features, respectively.
    @type X: numpy matrix of floats
    @return: the nonzero singular values and the corresponding left and right singular vectors of X. The singular vectors are contained in a r*1-matrix, where r is the number of nonzero singular values. 
    @rtype: a tuple of three numpy matrices"""
    evecs, svals, U = la.svd(X, full_matrices=0)
    evals = np.multiply(svals, svals)
    
    evecs = evecs[:, svals > 0]
    svals = svals[svals > 0]
    U = U[svals > 0]
    return evecs, svals, U


def eig_psd(K):
    """"Returns the reduced eigen decomposition of the kernel matrix K so that only the eigenvectors corresponding to the nonzero eigenvalues are returned.
    
    @param K: a positive semi-definite kernel matrix whose rows and columns are indexed by the datumns.
    @type K: numpy matrix of floats
    @return: the square roots of the nonzero eigenvalues and the corresponding eigenvectors of K. The square roots of the eigenvectors are contained in a r*1-matrix, where r is the number of nonzero eigenvalues. 
    @rtype: a tuple of two numpy matrices"""
    try:
        evals, evecs = la.eigh(K)
    except LinAlgError as e:
        print('Warning, caught a LinAlgError while eigen decomposing: ' + str(e))
        K = K + np.eye(K.shape[0]) * 0.0000000001
        evals, evecs = la.eigh(K)
    evecs = evecs[:, evals > 0.]
    evals = evals[evals > 0.]
    svals = np.sqrt(evals)
    return svals, evecs
  

