import numpy as np
from rlscore.learner.rls import RLS
from rlscore.measure import sqerror

from housing_data import load_housing


def train_rls():
    #Select regparam with leave-one-out cross-validation
    X_train, Y_train, X_test, Y_test = load_housing()
    learner = RLS(X_train, Y_train)
    best_regparam = None
    best_error = float("inf")
    #exponential grid of possible regparam values
    log_regparams = range(-15, 16)
    for log_regparam in log_regparams:
        regparam = 2.**log_regparam
        #RLS is re-trained with the new regparam, this
        #is very fast due to computational short-cut
        learner.solve(regparam)
        #Leave-one-out cross-validation predictions, this is fast due to
        #computational short-cut
        P_loo = learner.leave_one_out()
        e = sqerror(Y_train, P_loo)
        print("regparam 2**%d, loo-error %f" %(log_regparam, e))
        if e < best_error:
            best_error = e
            best_regparam = regparam
    learner.solve(best_regparam)
    P_test = learner.predict(X_test)
    print("best regparam %f with loo-error %f" %(best_regparam, best_error)) 
    print("test error %f" %sqerror(Y_test, P_test))

if __name__=="__main__":
    train_rls()
