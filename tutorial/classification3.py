import numpy as np
from rlscore.learner import RLS
from rlscore.measure import auc
from rlscore.reader import read_svmlight


def train_rls():
    X_train, Y_train, foo = read_svmlight("a1a.t")
    X_test, Y_test, foo = read_svmlight("a1a")
    lpo_aucs = []
    test_aucs = []
    for i in range(1000):
        X_small = X_train[i*30: i*30 + 30]
        Y_small = Y_train[i*30: i*30 + 30]
        pairs_start = []
        pairs_end = []
        for i in range(len(Y_small)):
            for j in range(len(Y_small)):
                if Y_small[i] == 1. and Y_small[j] == -1.:
                    pairs_start.append(i)
                    pairs_end.append(j)
        learner = RLS(X_small, Y_small)
        pairs_start = np.array(pairs_start)
        pairs_end = np.array(pairs_end)
        P_start, P_end = learner.leave_pair_out(pairs_start, pairs_end)
        lpo_a = np.mean(P_start > P_end + 0.5 * (P_start == P_end))
        P_test = learner.predict(X_test)
        test_a = auc(Y_test, P_test)
        lpo_aucs.append(lpo_a)
        test_aucs.append(test_a)
    print("mean lpo over auc over 1000 repetitions: %f" %np.mean(lpo_aucs))
    print("mean test auc over 1000 repetitions %f" %np.mean(test_aucs))

if __name__=="__main__":
    train_rls()
