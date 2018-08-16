from rlscore.learner import KronRLS
from rlscore.measure import cindex
import davis_data

def main():
    X1_train, X2_train, Y_train, X1_test, X2_test, Y_test = davis_data.settingC_split()
    learner = KronRLS(X1 = X1_train, X2 = X2_train, Y = Y_train)
    log_regparams = range(15, 35)
    for log_regparam in log_regparams:
        learner.solve(2.**log_regparam)
        P = learner.predict(X1_test, X2_test)
        perf = cindex(Y_test, P)
        print("regparam 2**%d, cindex %f" %(log_regparam, perf))

if __name__=="__main__":
    main()
