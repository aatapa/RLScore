from rlscore.learner import TwoStepRLS
from rlscore.measure import cindex
import davis_data

def main():
    X1_train, X2_train, Y_train, X1_test, X2_test, Y_test = davis_data.setting2_split()
    learner = TwoStepRLS(X1 = X1_train, X2 = X2_train, Y = Y_train, regparam1=1.0, regparam2=1.0)
    log_regparams1 = range(-8, -4)
    log_regparams2 = range(20,25)
    for log_regparam1 in log_regparams1:
        for log_regparam2 in log_regparams2:
            learner.solve(2.**log_regparam1, 2.**log_regparam2)
            P = learner.predict(X1_test, X2_test)
            perf = cindex(Y_test, P)
            print("regparam 2**%d 2**%d, test set cindex %f" %(log_regparam1, log_regparam2, perf))
            P = learner.leave_x1_out()
            perf = cindex(Y_train, P)
            print("regparam 2**%d 2**%d, leave-row-out cindex %f" %(log_regparam1, log_regparam2, perf))


if __name__=="__main__":
    main()
