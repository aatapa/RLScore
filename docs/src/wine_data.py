import numpy as np

def load_wine():
    np.random.seed(1)
    D = np.loadtxt("wine.data", delimiter=",")
    np.random.shuffle(D)  
    X = D[:,1:]
    Y = D[:,0]
    X_train = X[:100]
    Y_train = Y[:100]
    X_test = X[100:]
    Y_test = Y[100:]
    return X_train, Y_train, X_test, Y_test

def print_stats():
    X_train, Y_train, X_test, Y_test = load_wine()
    print("Wine data set characteristics")
    print("Training set: %d instances, %d features" %X_train.shape)
    print("Test set: %d instances, %d features" %X_test.shape)

if __name__=="__main__":
    print_stats()
