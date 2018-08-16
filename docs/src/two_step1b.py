import numpy as np

from rlscore.learner import TwoStepRLS
from rlscore.measure import cindex
import davis_data
from rlscore.utilities.cross_validation import random_folds

def main():
    X1, X2, Y = davis_data.load_davis()
    n = X1.shape[0]
    m = X2.shape[0]
    Y = Y.ravel(order='F')
    learner = TwoStepRLS(X1 = X1, X2 = X2, Y = Y, regparam1=1.0, regparam2=1.0)
    log_regparams1 = range(-8, -4)
    log_regparams2 = range(20,25)
    #Create random split to 5 folds for the drug-target pairs
    folds = random_folds(n*m, 5, seed=12345)
    #Map the indices back to (drug_indices, target_indices)
    folds = [np.unravel_index(fold, (n,m)) for fold in folds]
    for log_regparam1 in log_regparams1:
        for log_regparam2 in log_regparams2:
            learner.solve(2.**log_regparam1, 2.**log_regparam2)
            P = learner.in_sample_kfoldcv(folds)
            perf = cindex(Y, P)
            print("regparam 2**%d 2**%d, cindex %f" %(log_regparam1, log_regparam2, perf))



if __name__=="__main__":
    main()
