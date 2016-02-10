import numpy as np
from rlscore.learner.rls import LeaveOneOutRLS

from housing_data import load_housing

def mae(Y,P):
    Y = np.array(Y)
    P = np.array(P)
    error = np.mean(np.abs(Y-P))
    return error
mae.iserror = True
    

def train_rls():
    #Trains RLS with automatically selected regularization parameter
    X_train, Y_train, X_test, Y_test = load_housing()
    regparams = [2.**i for i in range(-15, 16)]
    learner = LeaveOneOutRLS(X_train, Y_train, regparams = regparams, measure=mae)
    loo_errors = learner.cv_performances
    P_test = learner.predict(X_test)
    print("leave-one-out mean absolute error " +str(loo_errors))
    print("chosen regparam %f" %learner.regparam)
    print("test mean absolute error %f" %mae(Y_test, P_test))

if __name__=="__main__":
    train_rls()
