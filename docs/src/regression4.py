from rlscore.learner import RLS
from rlscore.measure import sqerror

from housing_data import load_housing


def train_rls():
    #Selects both the gamma parameter for Gaussian kernel, and regparam with loocv
    X_train, Y_train, X_test, Y_test = load_housing()
    regparams = [2.**i for i in range(-15, 16)]
    gammas = regparams
    best_regparam = None
    best_gamma = None
    best_error = float("inf")
    for gamma in gammas:
        #New RLS is initialized for each kernel parameter
        learner = RLS(X_train, Y_train, kernel="GaussianKernel", gamma=gamma)
        for regparam in regparams:
            #RLS is re-trained with the new regparam, this
            #is very fast due to computational short-cut
            learner.solve(regparam)
            #Leave-one-out cross-validation predictions, this is fast due to
            #computational short-cut
            P_loo = learner.leave_one_out()
            e = sqerror(Y_train, P_loo)
            #print "regparam", regparam, "gamma", gamma, "loo-error", e
            if e < best_error:
                best_error = e
                best_regparam = regparam
                best_gamma = gamma
    learner = RLS(X_train, Y_train, regparam = best_regparam, kernel="GaussianKernel", gamma=best_gamma)
    P_test = learner.predict(X_test)
    print("best parameters gamma %f regparam %f" %(best_gamma, best_regparam))
    print("best leave-one-out error %f" %best_error)
    print("test error %f" %sqerror(Y_test, P_test))

if __name__=="__main__":
    train_rls()
