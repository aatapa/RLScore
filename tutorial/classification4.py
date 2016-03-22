from rlscore.learner import LeavePairOutRLS
from rlscore.measure import auc
from rlscore.reader import read_svmlight


def train_rls():
    X_train, Y_train, foo = read_svmlight("a1a.t")
    #subsample, leave-pair-out on whole data would take
    #a lot of time
    X_train = X_train[:1000]
    Y_train = Y_train[:1000]
    X_test, Y_test, foo = read_svmlight("a1a", X_train.shape[1])
    regparams = [2.**-5, 1., 2.**5]
    learner = LeavePairOutRLS(X_train, Y_train, regparams=regparams)
    print("best regparam %f" %learner.regparam)
    print("lpo auc " +str(learner.cv_performances))
    P_test = learner.predict(X_test)
    print("test auc %f" %auc(Y_test, P_test))

if __name__=="__main__":
    train_rls()
