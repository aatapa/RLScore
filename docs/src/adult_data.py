from rlscore.utilities.reader import read_svmlight

def print_stats():
    X_train, Y_train, foo = read_svmlight("a1a.t")
    X_test, Y_test, foo = read_svmlight("a1a")
    print("Adult data set characteristics")
    print("Training set: %d instances, %d features" %X_train.shape)
    print("Test set: %d instances, %d features" %X_test.shape)

if __name__=="__main__":
    print_stats()
