from rlscore.learner import LeaveOneOutRLS
from rlscore.measure import sqerror

from housing_data import load_housing


def train_rls():
    #Trains RLS with automatically selected regularization parameter
    X_train, Y_train, X_test, Y_test = load_housing()
    regparams = [2.**i for i in range(-15, 16)]
    learner = LeaveOneOutRLS(X_train, Y_train, regparams = regparams)
    loo_errors = learner.cv_performances
    P_test = learner.predict(X_test)
    print("leave-one-out errors " +str(loo_errors))
    print("chosen regparam %f" %learner.regparam)
    print("test error %f" %sqerror(Y_test, P_test))

if __name__=="__main__":
    train_rls()
