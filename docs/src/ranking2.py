from rlscore.learner import LeavePairOutRankRLS
from rlscore.measure import cindex

from housing_data import load_housing


def train_rls():
    #Trains RankRLS with automatically selected regularization parameter
    X_train, Y_train, X_test, Y_test = load_housing()
    regparams = [2.**i for i in range(-10, 10)]
    learner = LeavePairOutRankRLS(X_train, Y_train, regparams = regparams)
    loo_errors = learner.cv_performances
    P_test = learner.predict(X_test)
    print("leave-pair-out performances " +str(loo_errors))
    print("chosen regparam %f" %learner.regparam)
    print("test set cindex %f" %cindex(Y_test, P_test))


if __name__=="__main__":
    train_rls()
