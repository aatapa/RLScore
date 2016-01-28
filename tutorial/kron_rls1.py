from rlscore.learner.kron_rls import KronRLS
from rlscore.measure import cindex
import davis_data

def main():
    X1, X2, Y = davis_data.load_davis()
    Y = Y.ravel(order='F')
    learner = KronRLS(X1 = X1, X2 = X2, Y = Y)
    log_regparams = range(15, 35)
    for log_regparam in log_regparams:
        learner.solve(2.**log_regparam)
        P = learner.in_sample_loo()
        perf = cindex(Y, P)
        print("regparam 2**%d, cindex %f" %(log_regparam, perf))

if __name__=="__main__":
    main()
