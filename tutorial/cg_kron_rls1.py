from rlscore.learner import CGKronRLS
from rlscore.measure import cindex
import metz_data

class CallBack(object):

    def __init__(self, X1, X2, Y, row_inds, col_inds):
        self.X1 = X1
        self.X2 = X2
        self.Y = Y
        self.row_inds = row_inds
        self.col_inds = col_inds
        self.iter = 1

    def callback(self, learner):
        if self.iter%10 == 0:
            P = learner.predict(self.X1, self.X2, self.row_inds, self.col_inds)
            perf = cindex(self.Y, P)
            print("iteration %d cindex %f" %(self.iter, perf))
        self.iter += 1

    def finished(self, learner):
        pass
    
def main():
    XD, XT, train_drug_inds, train_target_inds, Y_train, test_drug_inds, test_target_inds, Y_test = metz_data.setting1_split()
    cb = CallBack(XD, XT, Y_test, test_drug_inds, test_target_inds)
    learner = CGKronRLS(X1 = XD, X2 = XT, Y=Y_train, label_row_inds = train_drug_inds, label_col_inds = train_target_inds, callback = cb, maxiter=1000)
    

if __name__=="__main__":
    main()
