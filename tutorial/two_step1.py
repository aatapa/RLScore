from rlscore.learner import TwoStepRLS
from rlscore.measure import cindex
import davis_data

def main():
    X1, X2, Y = davis_data.load_davis()
    Y = Y.ravel(order='F')
    learner = TwoStepRLS(X1 = X1, X2 = X2, Y = Y, regparam1=1.0, regparam2=1.0)
    log_regparams1 = range(-8, -4)
    log_regparams2 = range(20,25)
    for log_regparam1 in log_regparams1:
        for log_regparam2 in log_regparams2:
            learner.solve(2.**log_regparam1, 2.**log_regparam2)
            P = learner.in_sample_loo()
            perf = cindex(Y, P)
            print("regparam 2**%d 2**%d, cindex %f" %(log_regparam1, log_regparam2, perf))



if __name__=="__main__":
    main()
