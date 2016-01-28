import numpy as np
from rlscore.learner.rls import RLS
from rlscore.measure import sqerror
from rlscore.kernel.gaussian_kernel import GaussianKernel
from housing_data import load_housing
import random
random.seed(55)

def train_rls():
    X_train, Y_train, X_test, Y_test = load_housing()
    #select randomly 100 basis vectors
    indices = range(X_train.shape[0])
    indices = random.sample(indices, 100)
    basis_vectors = X_train[indices]
    kernel = GaussianKernel(basis_vectors, gamma=0.00003)
    K_train = kernel.getKM(X_train)
    K_rr = kernel.getKM(basis_vectors)
    K_test = kernel.getKM(X_test)
    learner = RLS(K_train, Y_train, basis_vectors = K_rr, kernel="PrecomputedKernel", regparam=0.0003)
    #Leave-one-out cross-validation predictions, this is fast due to
    #computational short-cut
    P_loo = learner.leave_one_out()
    #Test set predictions
    P_test = learner.predict(K_test)
    print("leave-one-out error %f" %sqerror(Y_train, P_loo))
    print("test error %f" %sqerror(Y_test, P_test))
    #Sanity check, can we do better than predicting mean of training labels?
    print("mean predictor %f" %sqerror(Y_test, np.ones(Y_test.shape)*np.mean(Y_train)))

if __name__=="__main__":
    train_rls()
