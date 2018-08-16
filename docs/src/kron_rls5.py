from rlscore.learner import KronRLS
from rlscore.measure import cindex
import davis_data

def main():
    X1_train, X2_train, Y_train, X1_test, X2_test, Y_test = davis_data.settingD_split()
    K1_train = X1_train.dot(X1_train.T)
    K2_train = X2_train.dot(X2_train.T)
    K1_test = X1_test.dot(X1_train.T)
    K2_test = X2_test.dot(X2_train.T)
    learner = KronRLS(K1 = K1_train, K2 = K2_train, Y = Y_train)
    log_regparams = range(15, 35)
    for log_regparam in log_regparams:
        learner.solve(2.**log_regparam)
        P = learner.predict(K1_test, K2_test)
        perf = cindex(Y_test, P)
        print("regparam 2**%d, cindex %f" %(log_regparam, perf))

if __name__=="__main__":
    main()
