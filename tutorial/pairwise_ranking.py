import numpy as np
from rlscore.learner.rankrls_with_pairwise_preferences import PPRankRLS
from rlscore.measure import cindex

from housing_data import load_housing
import random
random.seed(33)

def train_rls():
    #Trains RLS with default parameters (regparam=1.0, kernel='LinearKernel')
    X_train, Y_train, X_test, Y_test = load_housing()
    pairs_start = []
    pairs_end = []
    #Sample 1000 pairwise preferences from the data 
    trange = range(len(Y_train))
    while len(pairs_start) < 1000:
        ind0 = random.choice(trange)
        ind1 = random.choice(trange)
        if Y_train[ind0] > Y_train[ind1]:
            pairs_start.append(ind0)
            pairs_end.append(ind1)
        elif Y_train[ind0] < Y_train[ind1]:
            pairs_start.append(ind1)
            pairs_end.append(ind0)
    learner = PPRankRLS(X_train, pairs_start, pairs_end)
    #Test set predictions
    P_test = learner.predict(X_test)
    print("test cindex %f" %cindex(Y_test, P_test))

if __name__=="__main__":
    train_rls()
