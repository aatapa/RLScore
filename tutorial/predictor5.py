from rlscore.learner.kron_rls import KronRLS
import davis_data

def main():
    X1_train, X2_train, Y_train, X1_test, X2_test, Y_test = davis_data.setting4_split()
    learner = KronRLS(X1 = X1_train, X2 = X2_train, Y = Y_train, regparam=2.**30)
    predictor = learner.predictor
    print predictor.W
    #Predict labels for all X1_test - X2_test combinations)
    #Order: column-major: [(X1[0], X2[0]), (X1[1], X2[0])...]
    P = predictor.predict(X1_test, X2_test)
    print("Number of predictions: %d" %P.shape)
    print("three first predictions: " +str(P[:3]))
    x1_ind = [0,1,2]
    x2_ind = [0,0,0]
    P2 = predictor.predict(X1_test, X2_test, x1_ind, x2_ind)
    print("three first predictions again: " +str(P2))
    print("Number of coefficients %d x %d" %predictor.W.shape)

if __name__=="__main__":
    main()
