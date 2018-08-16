import numpy as np
from rlscore.learner import TwoStepRLS
from rlscore.measure import cindex
import davis_data

def main():
    X = np.loadtxt("drug-drug_similarities_2D.txt")
    Y = np.loadtxt("drug-drug_similarities_ECFP4.txt")
    Y = Y.ravel(order='F')
    Y = Y.ravel(order='F')
    K = np.dot(X, X)
    learner = TwoStepRLS(K1 = K, K2 = K, Y = Y, regparam1=1.0, regparam2=1.0)
    log_regparams = range(-10,0)
    for log_regparam in log_regparams:
        learner.solve(2.**log_regparam, 2.**log_regparam)
        P = learner.in_sample_loo_symmetric()
        perf = cindex(Y, P)
        print("regparam 2**%d, cindex %f" %(log_regparam, perf))



if __name__=="__main__":
    main()
