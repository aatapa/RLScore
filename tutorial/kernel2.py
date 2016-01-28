import numpy as np
from rlscore.learner.rls import RLS
from rlscore.measure import sqerror

from housing_data import load_housing


def train_rls():
    X_train, Y_train, X_test, Y_test = load_housing()
    learner = RLS(X_train, Y_train, kernel="GaussianKernel", regparam=0.0003, gamma=0.00003)
    #Leave-one-out cross-validation predictions, this is fast due to
    #computational short-cut
    P_loo = learner.leave_one_out()
    #Test set predictions
    P_test = learner.predict(X_test)
    print("leave-one-out error %f" %sqerror(Y_train, P_loo))
    print("test error %f" %sqerror(Y_test, P_test))
    #Sanity check, can we do better than predicting mean of training labels?
    print("mean predictor %f" %sqerror(Y_test, np.ones(Y_test.shape)*np.mean(Y_train)))

if __name__=="__main__":
    train_rls()
