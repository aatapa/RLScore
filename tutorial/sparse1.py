from rlscore.learner import CGRLS
from rlscore.measure import auc

from newsgroups_data import load_newsgroups

def train_rls():
    X_train, Y_train, X_test, Y_test = load_newsgroups()
    #CGRLS does not support multi-output learning, so we train
    #one classifier for the first column of Y. Multi-class learning
    #would be implemented by training one CGRLS for each column, and
    #taking the argmax of class predictions.
    predictions = []
    rls = CGRLS(X_train, Y_train[:,0], regparam= 100.0)
    P = rls.predict(X_test)
    perf = auc(Y_test[:,0], P)
    print "auc for task 1", perf 


if __name__=="__main__":
    train_rls()
