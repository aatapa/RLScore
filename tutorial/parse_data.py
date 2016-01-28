import numpy as np
from rlscore.reader import read_sparse
from rlscore.utilities.cross_validation import map_ids

def print_stats():
    X_train =  read_sparse("train_2000_x.txt")
    Y_train =  np.loadtxt("train_2000_y.txt")
    ids =  np.loadtxt("train_2000_qids.txt", dtype=int)
    folds = map_ids(ids)
    print("Parse data set characteristics")
    print("Training set: %d instances, %d features" %X_train.shape)
    print("Instances grouped into %d sentences" %len(folds))
    

if __name__=="__main__":
    print_stats()
