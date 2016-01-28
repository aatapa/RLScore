import numpy as np
import random

def load_metz():
    Y = np.loadtxt("known_drug-target_interaction_affinities_pKi__Metz_et_al.2011.txt")
    XD = np.loadtxt("drug-drug_similarities_2D__Metz_et_al.2011.txt")
    XT = np.loadtxt("target-target_similarities_WS_normalized__Metz_et_al.2011.txt")
    drug_inds, target_inds = np.where(np.isnan(Y)==False)
    Y = Y[drug_inds, target_inds]
    return XD, XT, Y, drug_inds, target_inds

def setting1_split():
    XD, XT, Y, drug_inds, target_inds = load_metz()
    random.seed(77)
    #random split to train/test, corresponds to setting 1
    ind = range(len(Y))
    random.shuffle(ind)
    train_ind = ind[:50000]
    test_ind = ind[50000:]
    train_drug_inds = drug_inds[train_ind]
    train_target_inds = target_inds[train_ind]
    Y_train = Y[train_ind]
    test_drug_inds = drug_inds[test_ind]
    test_target_inds = target_inds[test_ind]
    Y_test = Y[test_ind]
    return XD, XT, train_drug_inds, train_target_inds, Y_train, test_drug_inds, test_target_inds, Y_test

def setting2_split():
    XD, XT, Y, drug_inds, target_inds = load_metz()
    random.seed(77)
    #random split to train/test, corresponds to setting 2
    drows = range(XD.shape[0])
    random.shuffle(drows)
    train_drows = set(drows[:800])
    #test_drug_ind = set(drug_ind[800:])
    train_ind = []
    test_ind = []
    for i in range(len(drug_inds)):
        if drug_inds[i] in train_drows:
            train_ind.append(i)
        else:
            test_ind.append(i)
    train_drug_inds = drug_inds[train_ind]
    train_target_inds = target_inds[train_ind]
    Y_train = Y[train_ind]
    test_drug_inds = drug_inds[test_ind]
    test_target_inds = target_inds[test_ind]
    Y_test = Y[test_ind]
    return XD, XT, train_drug_inds, train_target_inds, Y_train, test_drug_inds, test_target_inds, Y_test

def setting3_split():
    XD, XT, Y, drug_inds, target_inds = load_metz()
    random.seed(77)
    #random split to train/test, corresponds to setting 3
    trows = range(XT.shape[0])
    random.shuffle(trows)
    train_trows = set(trows[:80])
    train_ind = []
    test_ind = []
    for i in range(len(target_inds)):
        if target_inds[i] in train_trows:
            train_ind.append(i)
        else:
            test_ind.append(i)
    train_drug_inds = drug_inds[train_ind]
    train_target_inds = target_inds[train_ind]
    Y_train = Y[train_ind]
    test_drug_inds = drug_inds[test_ind]
    test_target_inds = target_inds[test_ind]
    Y_test = Y[test_ind]
    return XD, XT, train_drug_inds, train_target_inds, Y_train, test_drug_inds, test_target_inds, Y_test


def setting4_split():
    XD, XT, Y, drug_inds, target_inds = load_metz()
    random.seed(77)
    #random split to train/test, corresponds to setting 4
    drows = range(XD.shape[0])
    random.shuffle(drows)
    train_drows = set(drows[:800])
    trows = range(XT.shape[0])
    random.shuffle(trows)
    train_trows = set(trows[:80])
    train_ind = []
    test_ind = []
    for i in range(len(target_inds)):
        if drug_inds[i] in train_drows and target_inds[i] in train_trows:
            train_ind.append(i)
        elif drug_inds[i] not in train_drows and target_inds[i] not in train_trows:
            test_ind.append(i)
    train_drug_inds = drug_inds[train_ind]
    train_target_inds = target_inds[train_ind]
    Y_train = Y[train_ind]
    test_drug_inds = drug_inds[test_ind]
    test_target_inds = target_inds[test_ind]
    Y_test = Y[test_ind]
    return XD, XT, train_drug_inds, train_target_inds, Y_train, test_drug_inds, test_target_inds, Y_test

if __name__=="__main__":
    XD, XT, Y, drug_inds, target_inds = load_metz()
    print("XD dimensions %d %d" %XD.shape)
    print("XT dimensions %d %d" %XT.shape)
    print("Labeled pairs %d, all possible pairs %d" %(len(Y), XD.shape[0]*XT.shape[0]))
