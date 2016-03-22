import numpy as np
import matplotlib.pyplot as plt
from rlscore.learner import RLS
from rlscore.measure import sqerror
from rlscore.utilities.cross_validation import map_ids
from rlscore.reader import read_sparse


def plot_rls():
    #Select regparam with k-fold cross-validation,
    #where instances related to a single sentence form
    #together a fold
    X_train =  read_sparse("train_2000_x.txt")
    Y_train =  np.loadtxt("train_2000_y.txt")
    X_test =  read_sparse("test_2000_x.txt", X_train.shape[1])
    Y_test =  np.loadtxt("test_2000_y.txt")
    #list of sentence ids
    ids =  np.loadtxt("train_2000_qids.txt")
    #mapped to a list of lists, where each list
    #contains indices for one fold
    folds = map_ids(ids)
    learner = RLS(X_train, Y_train)
    best_regparam = None
    best_error = float("inf")
    #exponential grid of possible regparam values
    log_regparams = range(-15, 16)
    kfold_errors = []
    loo_errors = []
    test_errors = []
    for log_regparam in log_regparams:
        regparam = 2.**log_regparam
        #RLS is re-trained with the new regparam, this
        #is very fast due to computational short-cut
        learner.solve(regparam)
        #K-fold cross-validation
        perfs = []
        for fold in folds:
            #computes holdout predictions, where instances
            #in fold are left out of training set
            P = learner.holdout(fold)
            perfs.append(sqerror(Y_train[fold], P))
        e_kfold = np.mean(perfs)
        kfold_errors.append(e_kfold)
        P_loo = learner.leave_one_out()
        e_loo = sqerror(Y_train, P_loo)
        loo_errors.append(e_loo)
        P_test = learner.predict(X_test)
        e_test = sqerror(Y_test, P_test)
        test_errors.append(e_test)
    plt.semilogy(log_regparams, loo_errors, label = "leave-one-out")
    plt.semilogy(log_regparams, kfold_errors, label = "leave-sentence-out")
    plt.semilogy(log_regparams, test_errors, label = "test error")
    plt.xlabel("$log_2(\lambda)$")
    plt.ylabel("mean squared error")
    plt.legend(loc=3)
    plt.show()


if __name__=="__main__":
    plot_rls()
