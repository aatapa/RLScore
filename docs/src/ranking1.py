from rlscore.learner import GlobalRankRLS
from rlscore.measure import cindex

from housing_data import load_housing


def train_rls():
    #Trains RLS with default parameters (regparam=1.0, kernel='LinearKernel')
    X_train, Y_train, X_test, Y_test = load_housing()
    learner = GlobalRankRLS(X_train, Y_train)
    #Test set predictions
    P_test = learner.predict(X_test)
    print("test cindex %f" %cindex(Y_test, P_test))

if __name__=="__main__":
    train_rls()
