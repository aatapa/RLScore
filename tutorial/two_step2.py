from rlscore.learner.two_step_rls import TwoStepRLS
from rlscore.measure import cindex
import davis_data

def main():
    X1_train, X2_train, Y_train, foo1, foo1, foo3 = davis_data.setting4_split()
    learner = TwoStepRLS(X1 = X1_train, X2 = X2_train, Y = Y_train, regparam1=1.0, regparam2=1.0)
    log_regparams1 = range(-8, -4)
    log_regparams2 = range(20,25)
    for log_regparam1 in log_regparams1:
        for log_regparam2 in log_regparams2:
            learner.solve(2.**log_regparam1, 2.**log_regparam2)
            P = learner.out_of_sample_loo()
            perf = cindex(Y_train, P)
            print("regparam 2**%d 2**%d, cindex %f" %(log_regparam1, log_regparam2, perf))

if __name__=="__main__":
    main()
