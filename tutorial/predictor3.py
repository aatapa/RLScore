from rlscore.learner import RLS

from housing_data import load_housing


def train_rls():
    X_train, Y_train, X_test, Y_test = load_housing()
    learner = RLS(X_train, Y_train, kernel="GaussianKernel", regparam=0.0003, gamma=0.00003)
    #This is how we make predictions
    P_test = learner.predict(X_test)
    #We can separate the predictor from learner
    predictor = learner.predictor
    #And do the same predictions
    P_test = predictor.predict(X_test)
    #Let's get the coefficients of the predictor
    A = predictor.A
    print("A-coefficients " +str(A))
    print("number of coefficients %d" %len(A))

if __name__=="__main__":
    train_rls()
