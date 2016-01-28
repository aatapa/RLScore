import numpy as np
from rlscore.learner.greedy_rls import GreedyRLS
from rlscore.measure import sqerror

from housing_data import load_housing

class Callback(object):

    def __init__(self, X_test, Y_test):
        self.X_test = X_test
        self.Y_test = Y_test
        self.iteration = 0

    def callback(self, learner):
        self.iteration += 1
        P = learner.predict(self.X_test)
        e = sqerror(self.Y_test, P)
        print("Features selected %d, test error %f" %(self.iteration, e))

    def finished(self, learner):
        pass

def train_rls():
    X_train, Y_train, X_test, Y_test = load_housing()
    cb = Callback(X_test, Y_test)
    learner = GreedyRLS(X_train, Y_train, 13, callbackfun = cb)
    #Test set predictions
    P_test = learner.predict(X_test)
    print("test error %f" %sqerror(Y_test, P_test))
    print("Selected features " +str(learner.selected))

if __name__=="__main__":
    train_rls()
