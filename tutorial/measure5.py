import numpy as np
from rlscore.learner.rls import LeaveOneOutRLS
from rlscore.measure import cindex

from housing_data import load_housing


def train_rls():
    #Trains RLS with automatically selected regularization parameter
    X_train, Y_train, X_test, Y_test = load_housing()
    regparams = [2.**i for i in range(-15, 16)]
    learner = LeaveOneOutRLS(X_train, Y_train, regparams = regparams, measure=cindex)
    loo_errors = learner.cv_performances
    P_test = learner.predict(X_test)
    print("leave-one-out cindex " +str(loo_errors))
    print("chosen regparam %f" %learner.regparam)
    print("test cindex %f" %cindex(Y_test, P_test))

if __name__=="__main__":
    train_rls()
