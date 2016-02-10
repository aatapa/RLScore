from rlscore.learner import RLS
from rlscore.measure import accuracy
from rlscore.reader import read_svmlight


def train_rls():
    X_train, Y_train, foo = read_svmlight("a1a.t")
    X_test, Y_test, foo = read_svmlight("a1a")
    learner = RLS(X_train, Y_train)
    best_regparam = None
    best_accuracy = 0.
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
        acc = accuracy(Y_train, P_loo)
        print("regparam 2**%d, loo-accuracy %f" %(log_regparam, acc))
        if acc > best_accuracy:
            best_accuracy = acc
            best_regparam = regparam
    learner.solve(best_regparam)
    P_test = learner.predict(X_test)
    print("best regparam %f with loo-accuracy %f" %(best_regparam, best_accuracy)) 
    print("test set accuracy %f" %accuracy(Y_test, P_test))

if __name__=="__main__":
    train_rls()
