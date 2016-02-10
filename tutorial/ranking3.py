import numpy as np
from rlscore.learner import GlobalRankRLS
from rlscore.measure import cindex
from rlscore.utilities.cross_validation import random_folds

from housing_data import load_housing


def train_rls():
    #Trains RLS with default parameters (regparam=1.0, kernel='LinearKernel')
    X_train, Y_train, X_test, Y_test = load_housing()
    #generate fold partition, arguments: train_size, k, random_seed
    folds = random_folds(len(Y_train), 5, 10)
    learner = GlobalRankRLS(X_train, Y_train)
    perfs = []
    for fold in folds:
        P = learner.holdout(fold)
        c = cindex(Y_train[fold], P)
        perfs.append(c)
    perf = np.mean(perfs)
    print("5-fold cross-validation cindex %f" %perf)
    P_test = learner.predict(X_test)
    print("test cindex %f" %cindex(Y_test, P_test))

if __name__=="__main__":
    train_rls()
