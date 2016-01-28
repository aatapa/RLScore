import numpy as np
from rlscore.learner.greedy_rls import GreedyRLS
from rlscore.measure import sqerror

from housing_data import load_housing


def train_rls():
    X_train, Y_train, X_test, Y_test = load_housing()
    #we select 5 features
    learner = GreedyRLS(X_train, Y_train, 5)
    #Test set predictions
    P_test = learner.predict(X_test)
    print("test error %f" %sqerror(Y_test, P_test))
    print("Selected features " +str(learner.selected))

if __name__=="__main__":
    train_rls()
