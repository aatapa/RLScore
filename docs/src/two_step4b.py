from rlscore.learner import TwoStepRLS
from rlscore.measure import cindex
import davis_data
from rlscore.utilities.cross_validation import random_folds

def main():
    X1_train, X2_train, Y_train, X1_test, X2_test, Y_test = davis_data.settingD_split()
    n = X1_train.shape[0]
    m = X2_train.shape[0]
    learner = TwoStepRLS(X1 = X1_train, X2 = X2_train, Y = Y_train, regparam1=1.0, regparam2=1.0)
    log_regparams1 = range(-8, -4)
    log_regparams2 = range(20,25)
    #Create random split to 5 folds for both drugs and targets
    drug_folds = random_folds(n, 5, seed=123)
    target_folds = random_folds(m, 5, seed=456)
    for log_regparam1 in log_regparams1:
        for log_regparam2 in log_regparams2:
            learner.solve(2.**log_regparam1, 2.**log_regparam2)
            P = learner.predict(X1_test, X2_test)
            perf = cindex(Y_test, P)
            print("regparam 2**%d 2**%d, test set cindex %f" %(log_regparam1, log_regparam2, perf))
            P = learner.out_of_sample_kfold_cv(drug_folds, target_folds)
            perf = cindex(Y_train, P)
            print("regparam 2**%d 2**%d, out-of-sample loo cindex %f" %(log_regparam1, log_regparam2, perf))

if __name__=="__main__":
    main()
