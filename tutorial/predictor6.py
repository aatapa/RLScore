from rlscore.learner import KronRLS
from rlscore.kernel import GaussianKernel
import davis_data

def main():
    X1_train, X2_train, Y_train, X1_test, X2_test, Y_test = davis_data.setting4_split()
    kernel1 = GaussianKernel(X1_train, gamma=0.01)
    kernel2 = GaussianKernel(X2_train, gamma=10**-9)
    K1_train = kernel1.getKM(X1_train)
    K1_test = kernel1.getKM(X1_test)
    K2_train = kernel2.getKM(X2_train)
    K2_test = kernel2.getKM(X2_test)
    learner = KronRLS(K1 = K1_train, K2 = K2_train, Y = Y_train, regparam=2**-5)
    predictor = learner.predictor
    P = predictor.predict(K1_test, K2_test)
    print("Number of predictions: %d" %P.shape)
    print("three first predictions: " +str(P[:3]))
    x1_ind = [0,1,2]
    x2_ind = [0,0,0]
    P2 = predictor.predict(X1_test, X2_test, x1_ind, x2_ind)
    print("three first predictions again: " +str(P2))
    print("Number of coefficients %d" %predictor.A.shape)

if __name__=="__main__":
    main()
