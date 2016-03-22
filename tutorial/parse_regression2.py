import numpy as np
from rlscore.learner import RLS
from rlscore.measure import sqerror
from rlscore.utilities.cross_validation import map_ids
from rlscore.reader import read_sparse


def train_rls():
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
    for log_regparam in log_regparams:
        regparam = 2.**log_regparam
        #RLS is re-trained with the new regparam, this
        #is very fast due to computational short-cut
        learner.solve(regparam)
        #K-fold cross-validation
        P = np.zeros(Y_train.shape)
        for fold in folds:
            #computes holdout predictions, where instances
            #in fold are left out of training set
            P[fold] = learner.holdout(fold)
        e = sqerror(Y_train, P)
        print("regparam 2**%d, k-fold error %f" %(log_regparam, e))
        if e < best_error:
            best_error = e
            best_regparam = regparam
    learner.solve(best_regparam)
    P_test = learner.predict(X_test)
    print("best regparam %f k-fold error %f" %(best_regparam, best_error))
    print("test error %f" %sqerror(Y_test, P_test))

if __name__=="__main__":
    train_rls()
