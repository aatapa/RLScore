import numpy as np
from rlscore.learner.rls import RLS
from rlscore.measure import sqerror

from housing_data import load_housing
import random
random.seed(55)

def train_rls():
    X_train, Y_train, X_test, Y_test = load_housing()
    #select randomly 20 basis vectors
    indices = range(X_train.shape[0])
    indices = random.sample(indices, 20)
    basis_vectors = X_train[indices]    
    learner = RLS(X_train, Y_train, basis_vectors = basis_vectors, kernel="GaussianKernel", regparam=0.0003, gamma=0.00003)
    #Test set predictions
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
