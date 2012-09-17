import numpy as np


def ova_accuracy(Y, P):
    """One-vs-all classification accuracy for multi-class problems.
    
    Computes the accuracy for a one-versus-all decomposed classification
    problem. Each column in Y and P correspond to one possible class label.
    On each row, exactly one column in Y is 1, all the rest must be -1. The
    prediction for the i:th example is computed by taking the argmax over
    the indices of row i in P. 
    
    Parameters
    ----------
    Y: {array-like}, shape = [n_samples] or [n_samples, n_classes]
        Correct labels, must belong to set {-1,1}, with exactly
        one 1 on each row.
    P: {array-like}, shape = [n_samples] or [n_samples, n_classes]
        Predicted labels, can be any real numbers.
    
    Returns
    -------
    accuracy: float
        number between 0 and 1
    """
    assert Y.shape == P.shape
    correct = 0
    for i in range(Y.shape[0]):
        largest_pred = None
        predicted = None
        true = None
        for j in range(Y.shape[1]):
            if Y[i,j] == 1:
                true = j
            if (not largest_pred) or  (P[i,j]>largest_pred):
                largest_pred = P[i,j]
                predicted = j
        if true == predicted:
            correct += 1
    perf = float(correct)/float(Y.shape[0])
    return perf
ova_accuracy.iserror = False
