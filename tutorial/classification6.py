import numpy as np
from rlscore.learner.rls import LeaveOneOutRLS
from rlscore.measure import ova_accuracy
from wine_data import load_wine
from rlscore.utilities.multiclass import to_one_vs_all

def train_rls():
    X_train, Y_train, X_test, Y_test = load_wine()
    #Map labels from set {1,2,3} to one-vs-all encoding
    Y_train = to_one_vs_all(Y_train)
    Y_test = to_one_vs_all(Y_test)
    regparams = [2.**i for i in range(-15, 16)]
    learner = LeaveOneOutRLS(X_train, Y_train, regparams=regparams, measure=ova_accuracy)
    P_test = learner.predict(X_test)
    #ova_accuracy computes one-vs-all classification accuracy directly between transformed
    #class label matrix, and a matrix of predictions, where each column corresponds to a class
    print("test set accuracy %f" %ova_accuracy(Y_test, P_test))

if __name__=="__main__":
    train_rls()
