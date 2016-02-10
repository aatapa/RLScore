from rlscore.learner import GreedyRLS

from housing_data import load_housing


def train_rls():
    #Trains RLS with default parameters (regparam=1.0, kernel='LinearKernel')
    X_train, Y_train, X_test, Y_test = load_housing()
    learner = GreedyRLS(X_train, Y_train, 5)
    #This is how we make predictions
    P_test = learner.predict(X_test)
    #We can separate the predictor from learner
    predictor = learner.predictor
    #And do the same predictions
    P_test = predictor.predict(X_test)
    #Let's get the coefficients of the predictor
    w = predictor.W
    b = predictor.b
    print("number of coefficients %d" %len(w))
    print("w-coefficients " +str(w))
    print("bias term %f" %b)

if __name__=="__main__":
    train_rls()
