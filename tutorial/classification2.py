import numpy as np
from rlscore.learner import RLS
from rlscore.measure import auc
from rlscore.reader import read_svmlight


def train_rls():
    X_train, Y_train, foo = read_svmlight("a1a.t")
    X_test, Y_test, foo = read_svmlight("a1a")
    loo_aucs = []
    test_aucs = []
    for i in range(1000):
        X_small = X_train[i*30: i*30 + 30]
        Y_small = Y_train[i*30: i*30 + 30]
        learner = RLS(X_small, Y_small)
        P_loo = learner.leave_one_out()
        loo_a = auc(Y_small, P_loo)
        P_test = learner.predict(X_test)
        test_a = auc(Y_test, P_test)
        loo_aucs.append(loo_a)
        test_aucs.append(test_a)
    print("mean loo auc over 1000 repetitions %f" %np.mean(loo_aucs))
    print("mean test auc over 1000 repetitions %f" %np.mean(test_aucs))

if __name__=="__main__":
    train_rls()
