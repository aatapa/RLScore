import numpy as np

def to_one_vs_all(Y):
    #maps vector of class labels from range {0,..,classcount} to matrix,
    #of size ssize x classes
    Y = np.array(Y)
    Y_ova = -1. * np.ones((Y.shape[0], np.max(Y+1)))
    for i in range(len(Y)):
        Y_ova[i, Y[i]] = 1
    return Y_ova

def from_one_vs_all(Y):
    #maps one-vs-all encoding or predictions to classes ranging 1...classcount
    return np.argmax(Y, axis=1)
