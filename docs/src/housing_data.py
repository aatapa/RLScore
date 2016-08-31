import numpy as np

def load_housing():
    np.random.seed(1)
    D = np.loadtxt("housing.data")
    np.random.shuffle(D)
    X = D[:,:-1]
    Y = D[:,-1]
    X_train = X[:250]
    Y_train = Y[:250]
    X_test = X[250:]
    Y_test = Y[250:]
    return X_train, Y_train, X_test, Y_test

def print_stats():
    X_train, Y_train, X_test, Y_test = load_housing()
    print("Housing data set characteristics")
    print("Training set: %d instances, %d features" %X_train.shape)
    print("Test set: %d instances, %d features" %X_test.shape)

if __name__ == "__main__":
    print_stats()


