from rlscore.learner import KfoldRankRLS
from rlscore.measure import cindex
from rlscore.utilities.cross_validation import random_folds

from housing_data import load_housing


def train_rls():
    #Trains RankRLS with automatically selected regularization parameter
    X_train, Y_train, X_test, Y_test = load_housing()
    #generate fold partition, arguments: train_size, k, random_seed
    folds = random_folds(len(Y_train), 5, 10)
    regparams = [2.**i for i in range(-10, 10)]
    learner = KfoldRankRLS(X_train, Y_train, folds = folds, regparams = regparams, measure=cindex)
    kfold_perfs = learner.cv_performances
    P_test = learner.predict(X_test)
    print("kfold performances " +str(kfold_perfs))
    print("chosen regparam %f" %learner.regparam)
    print("test set cindex %f" %cindex(Y_test, P_test))

if __name__=="__main__":
    train_rls()
