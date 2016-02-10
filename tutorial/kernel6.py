import numpy as np
from rlscore.learner import RLS
from rlscore.measure import sqerror
from rlscore.kernel import GaussianKernel
from housing_data import load_housing


def train_rls():
    X_train, Y_train, X_test, Y_test = load_housing()
    kernel = GaussianKernel(X_train, gamma = 0.00003)
    K_train = kernel.getKM(X_train)
    K_test = kernel.getKM(X_test)
    learner = RLS(K_train, Y_train, kernel="PrecomputedKernel", regparam=0.0003)
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
