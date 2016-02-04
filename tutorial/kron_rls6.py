from rlscore.learner.kron_rls import KronRLS
from rlscore.measure import cindex
from rlscore.kernel.gaussian_kernel import GaussianKernel
import davis_data

def main():
    X1_train, X2_train, Y_train, X1_test, X2_test, Y_test = davis_data.setting4_split()
    kernel1 = GaussianKernel(X1_train, gamma=0.01)
    kernel2 = GaussianKernel(X2_train, gamma=10**-9)
    K1_train = kernel1.getKM(X1_train)
    K1_test = kernel1.getKM(X1_test)
    K2_train = kernel2.getKM(X2_train)
    K2_test = kernel2.getKM(X2_test)
    learner = KronRLS(K1 = K1_train, K2 = K2_train, Y = Y_train)
    log_regparams = range(-15, 15)
    for log_regparam in log_regparams:
        learner.solve(2.**log_regparam)
        P = learner.predict(K1_test, K2_test)
        perf = cindex(Y_test, P)
        print("regparam 2**%d, cindex %f" %(log_regparam, perf))

if __name__=="__main__":
    main()
