import numpy as np
from mnist import MNIST
from rlscore.learner.greedy_rls import GreedyRLS
from rlscore.measure import ova_accuracy

def ova(Y):
    #maps range 0,...,classcount -1 to one-vs-all encoding
    Y_ova = -1. * np.ones((len(Y), np.max(Y)+1))
    for i in range(len(Y)):
        Y_ova[i, Y[i]] = 1
    return Y_ova

class Callback(object):

    def __init__(self, X_test, Y_test):
        self.X_test = X_test
        self.Y_test = Y_test
        self.iteration = 0

    def callback(self, learner):
        self.iteration += 1
        P = learner.predict(self.X_test)
        acc = ova_accuracy(self.Y_test, P)
        print("Features selected %d, accuracy %f" %(self.iteration, acc))

    def finished(self, learner):
        pass

def train_rls():
    mndata = MNIST("./data")
    X_train, Y_train = mndata.load_training()
    X_test, Y_test = mndata.load_testing()
    X_train, X_test = np.array(X_train), np.array(X_test)
    print X_train.shape
    assert False
    #One-vs-all mapping
    Y_train = ova(Y_train)
    Y_test = ova(Y_test)
    #Train greedy RLS, select 10 features
    cb = Callback(X_test, Y_test)
    learner = GreedyRLS(X_train, Y_train, 50, callbackfun=cb)
    print("Selected features " +str(learner.selected))

if __name__=="__main__":
    train_rls()
