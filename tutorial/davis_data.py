import numpy as np
import random

def load_davis():
    Y = np.loadtxt("drug-target_interaction_affinities_Kd__Davis_et_al.2011.txt")
    XD = np.loadtxt("drug-drug_similarities_2D.txt")
    XT = np.loadtxt("target-target_similarities_WS_normalized.txt")    
    return XD, XT, Y

def setting2_split():
    random.seed(10)
    XD, XT, Y = load_davis()
    drug_ind = range(Y.shape[0])
    random.shuffle(drug_ind)
    train_drug_ind = drug_ind[:40]
    test_drug_ind = drug_ind[40:]
    #Setting 2: split according to drugs
    Y_train = Y[train_drug_ind]
    Y_test = Y[test_drug_ind]
    Y_train = Y_train.ravel(order='F')
    Y_test = Y_test.ravel(order='F')
    XD_train = XD[train_drug_ind]
    XT_train = XT
    XD_test = XD[test_drug_ind]
    XT_test = XT
    return XD_train, XT_train, Y_train, XD_test, XT_test, Y_test   

def setting3_split():
    random.seed(10)
    XD, XT, Y = load_davis()
    drug_ind = range(Y.shape[0])
    target_ind = range(Y.shape[1])
    random.shuffle(target_ind)
    train_target_ind = target_ind[:300]
    test_target_ind = target_ind[300:]
    #Setting 3: split according to targets
    Y_train = Y[:, train_target_ind]
    Y_test = Y[:, test_target_ind]
    Y_train = Y_train.ravel(order='F')
    Y_test = Y_test.ravel(order='F')
    XD_train = XD
    XT_train = XT[train_target_ind]
    XD_test = XD
    XT_test = XT[test_target_ind]
    return XD_train, XT_train, Y_train, XD_test, XT_test, Y_test  

def setting4_split():
    random.seed(10)
    XD, XT, Y = load_davis()
    drug_ind = range(Y.shape[0])
    target_ind = range(Y.shape[1])
    random.shuffle(drug_ind)
    random.shuffle(target_ind)
    train_drug_ind = drug_ind[:40]
    test_drug_ind = drug_ind[40:]
    train_target_ind = target_ind[:300]
    test_target_ind = target_ind[300:]
    #Setting 4: ensure that d,t pairs do not overlap between
    #training and test set
    Y_train = Y[np.ix_(train_drug_ind, train_target_ind)]
    Y_test = Y[np.ix_(test_drug_ind, test_target_ind)]
    Y_train = Y_train.ravel(order='F')
    Y_test = Y_test.ravel(order='F')
    XD_train = XD[train_drug_ind]
    XT_train = XT[train_target_ind]
    XD_test = XD[test_drug_ind]
    XT_test = XT[test_target_ind]
    return XD_train, XT_train, Y_train, XD_test, XT_test, Y_test    

if __name__=="__main__":
    XD, XT, Y = load_davis()
    print("Y dimensions %d %d" %Y.shape)
    print("XD dimensions %d %d" %XD.shape)
    print("XT dimensions %d %d" %XT.shape)
    print("drug-target pairs: %d" %(Y.shape[0]*Y.shape[1]))
