from rlscore.learner import RLS
from rlscore.measure import auc
from rlscore.utilities.reader import read_svmlight


def train_rls():
    X_train, Y_train, foo = read_svmlight("a1a.t")
    X_test, Y_test, foo = read_svmlight("a1a", X_train.shape[1])
    learner = RLS(X_train, Y_train)
    best_regparam = None
    best_auc = 0.
    #exponential grid of possible regparam values
    log_regparams = range(-15, 16)
    for log_regparam in log_regparams:
        regparam = 2.**log_regparam
        #RLS is re-trained with the new regparam, this
        #is very fast due to computational short-cut
        learner.solve(regparam)
        #Leave-one-out cross-validation predictions, this is fast due to
        #computational short-cut
        P_loo = learner.leave_one_out()
        acc = auc(Y_train, P_loo)
        print("regparam 2**%d, loo-auc %f" %(log_regparam, acc))
        if acc > best_auc:
            best_auc = acc
            best_regparam = regparam
    learner.solve(best_regparam)
    P_test = learner.predict(X_test)
    print("best regparam %f with loo-auc %f" %(best_regparam, best_auc)) 
    print("test set auc %f" %auc(Y_test, P_test))

if __name__=="__main__":
    train_rls()
