import numpy as np
from rlscore.learner import KfoldRankRLS
from rlscore.measure import cindex
from rlscore.utilities.cross_validation import random_folds

from housing_data import load_housing


def train_rls():
    #Selects both the gamma parameter for Gaussian kernel, and regparam with kfoldcv
    X_train, Y_train, X_test, Y_test = load_housing()
    folds = random_folds(len(Y_train), 5, 10)
    regparams = [2.**i for i in range(-15, 16)]
    gammas = regparams
    best_regparam = None
    best_gamma = None
    best_perf = 0.
    best_learner = None
    for gamma in gammas:
        #New RLS is initialized for each kernel parameter
        learner = KfoldRankRLS(X_train, Y_train, kernel = "GaussianKernel", folds = folds, gamma = gamma, regparams = regparams, measure=cindex)
        perf = np.max(learner.cv_performances)
        if perf > best_perf:
            best_perf = perf
            best_regparam = learner.regparam
            best_gamma = gamma
            best_learner = learner
    P_test = best_learner.predict(X_test)
    print("best parameters gamma %f regparam %f" %(best_gamma, best_regparam))
    print("best kfoldcv cindex %f" %best_perf)
    print("test cindex %f" %cindex(Y_test, P_test))

if __name__=="__main__":
    train_rls()
