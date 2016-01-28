import numpy as np
from rlscore.learner.rls import RLS
from rlscore.measure import sqerror

from housing_data import load_housing


def train_rls():
    #Trains RLS with a precomputed kernel matrix
    X_train, Y_train, X_test, Y_test = load_housing()
    #Minor techincal detail: adding 1.0 simulates the effect of adding a
    #constant valued bias feature, as is done by 'LinearKernel' by deafault
    K_train = np.dot(X_train, X_train.T) + 1.0
    K_test = np.dot(X_test, X_train.T) + 1.0
    learner = RLS(K_train, Y_train, kernel="PrecomputedKernel")
    #Leave-one-out cross-validation predictions, this is fast due to
    #computational short-cut
    P_loo = learner.leave_one_out()
    #Test set predictions
    P_test = learner.predict(K_test)
    print("leave-one-out error %f" %sqerror(Y_train, P_loo))
    print("test error %f" %sqerror(Y_test, P_test))
    #Sanity check, can we do better than predicting mean of training labels?
    print("mean predictor %f" %sqerror(Y_test, np.ones(Y_test.shape)*np.mean(Y_train)))

if __name__=="__main__":
    train_rls()
