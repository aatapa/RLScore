from scipy import sparse
from numpy import float64, loadtxt
import numpy as np
from cPickle import load as cPickleload

def read_folds(fname):
    """ Reads a list of fold index lists.
    
    Format: let the training set indices range from 0... n_samples-1. Each line
    in a fold file should contain a subset of these indices corresponding to a
    single fold. For example, let n_samples = 11, then:
    
    0 3 4 8
    1 5 9 10
    2 6 7
    
    would correspond to a fold-file with three folds, with first and second fold
    containing 4, and last one 3 instances. The reader would return the list
    
    [[0,3,4,8],[1,5,9,10],[2,6,7]]
    
    Parameters
    ----------
    fname : string
        input file name
        
    Returns
    -------
    folds : a list of lists, each containing the indices corresponding to a single fold
    """
    f = open(fname)
    folds = []
    for i, line in enumerate(f):
        #We allow comments starting with #
        cstart = line.find("#")
        if cstart != -1:
            line = line[:cstart]
        fold = []
        foldset = set([])
        line = line.strip().split()
        for x in line:
            try:
                index = int(x)
            except ValueError:
                raise Exception("Error when reading in fold file: malformed index on line %d in the fold file: %s" % (i + 1, x))
            if index < 0:
                raise Exception("Error when reading in fold file: negative index on line %d in the fold file: %d" % (i + 1, index))
            if index in foldset:
                raise Exception("Error when reading in fold file: duplicate index on line %d in the fold file: %d" % (i + 1, index + 1))
            fold.append(index)
            foldset.add(index)
        folds.append(fold)
    f.close()
    return folds



def read_sparse(fname):
    """Reads in a sparse n x m matrix from a file with n rows.
    
    Format is of the type 0:1.5 3:4.2 7:1.1 ...
    with each line containing index:value pairs with indices
    ranging from 0...n_features-1, and only indices with non-zero values
    being present in the file.
    
    Parameters
    ----------
    fname : string
        input file name
        
    Returns
    -------
    X: sparse matrix (csr)
    """
    #each row represents an instance, each column a feature
    f = open(fname)
    rows = []
    columns = []
    values = []
    linecounter = 0
    feaspace_dim = 0
    for line in f:
        linecounter += 1
        #Empty lines and commented lines are passed over
        if len(line.strip()) == 0 or line[0] == '#':
            print "Warning: no inputs on line %d" % linecounter
            continue
        line = line.split("#",1)
        attributes = line[0].split()
        previous = -1
        #Attributes indices must be positive integers in an ascending order,
        #and the values must be real numbers.
        for att_val in attributes:
            if len(att_val.split(":")) != 2:
                raise Exception("Error when reading in feature file: feature:value pair %s on line %d is not well-formed\n" % (att_val, linecounter))
            index, value = att_val.split(":")
            try:
                index = int(index)
                value = float(value)
                if value != 0.:
                    columns.append(index)
                    rows.append(linecounter-1)
                    values.append(value)
            except ValueError:
                raise Exception("Error when reading in feature file: feature:value pair %s on line %d is not well-formed\n" % (att_val, linecounter))
            if not index > previous:
                raise Exception("Error when reading in feature file: line %d features must be in ascending order\n" % (linecounter))
            previous = index
            if index+1 > feaspace_dim:
                feaspace_dim = index+1
    #That's all folks
    row_size = linecounter
    col_size = feaspace_dim
    X = sparse.coo_matrix((values,(rows,columns)),(row_size, col_size), dtype=float64)
    X = X.tocsr()
    f.close()
    return X


def read_svmlight(fname):
    """ Loads examples from an SVM-light format data file. The
    file contains attributes, one label per example and optionally qids.
    
    Parameters
    ----------
    fname : string
        input file name
        
    Returns
    -------
    tuple : ['spmatrix': X, 'matrix':Y, 'qids':Q]
    
    X : sparse csc_matrix, shape = [n_samples, n_features]
    Y : ndarray, shape = [n_samples, n_labels]
    Q : list of n_queries index lists
    """
    f = open(fname)
    #some interesting statistics are calculated
    labelcount = None
    linecounter = 0
    feaspace_dim = 0
    #Features, labels, comments and possibly qids are later returned to caller
    #The indexing, with respect to the instances, is the same in all the lists.
    qids = None
     
    rows = []
    columns = []
    values = []
    
    all_outputs = []
    
    #Each line in the source represents an instance
    for linenumber, line in enumerate(f):
        if line[0] == "#" or line.strip() == "":
            continue
        linecounter += 1
        line = line.split('#')
        line = line[0].split()
        labels = line.pop(0)
        if line[0].startswith("qid:"):
            qid = line.pop(0)[4:]
            if qids == None:
                if linecounter > 1:
                    raise Exception("Error when reading in SVMLight file: Line %d has a qid, previous lines did not have qids defined" % (linenumber))   
                else:
                    qids = [qid]
            else:
                qids.append(qid)
        else:
            if qids != None:
                raise Exception("Error when reading in SVMLight file: Line %d has no qid, previous lines had qids defined" % (linenumber))
        attributes = line
        #Multiple labels are allowed, but each instance must have the
        #same amount of them. Labels must be real numbers.
        labels = labels.split("|")
        if labelcount == None:
            labelcount = len(labels)
        #Check that the number of labels is the same for all instances
        #and that the labels are real valued numbers.
        else:
            if labelcount != len(labels):
                raise Exception("Error when reading in SVMLight file: Number of labels assigned to instances differs.\n First instance had %d labels whereas instance on line %d has %d labels\n" % (labelcount, linenumber, len(labels)))
        label_list = []
        #We check that the labels are real numbers and gather them
        for label in labels:
            try:
                label = float(label)
                label_list.append(label)
            except ValueError:
                raise Exception("Error when reading in SVMLight file: label %s on line %d not a real number\n" % (label, linenumber))
        all_outputs.append(label_list)
        previous = 0
        #Attributes indices must be positive integers in an ascending order,
        #and the values must be real numbers.
        for att_val in attributes:
            if len(att_val.split(":")) != 2:
                raise Exception("Error when reading in SVMLight file: feature:value pair %s on line %d is not well-formed\n" % (att_val, linenumber))
            index, value = att_val.split(":")
            try:
                index = int(index)
                value = float(value)
                if value != 0.:
                    #rows.append(index-1)
                    rows.append(linecounter-1)
                    #columns.append(linecounter-1)
                    columns.append(index-1)
                    values.append(value)
            except ValueError:
                raise Exception("Error when reading in SVMLight file: feature:value pair %s on line %d is not well-formed\n" % (att_val, linecounter))
            if not index > previous:
                raise Exception("Error when reading in SVMLight file: line %d features must be in ascending order\n" % (linecounter))
            previous = index
            if index > feaspace_dim:
                feaspace_dim = index
    #X = sparse.coo_matrix((values,(rows,columns)),(feaspace_dim, linecounter), dtype=float64)
    X = sparse.coo_matrix((values,(rows,columns)),(linecounter, feaspace_dim), dtype=float64)
    X = X.tocsr()
    Y = np.array(all_outputs)
    if not qids == None:
        Q = mapQids(qids)
    else:
        Q = None
    f.close()
    #return {"spmatrix":X, "matrix":Y, "qids":Q}
    return X, Y, Q


def read_preferences(fname):
    """Reads a pairwise preferences file, used typically with ranking
    
    Parameters
    ----------
    fname : string
        input file name
        
    Returns
    -------
    data : n x 2 -dimensional numpy array containing pairwise preferences one pair per row, i.e. the data point corresponding to the first index is preferred over the data point corresponding to the second index. 
    """
    data = np.loadtxt(fname)
    if data.shape[1] != 2:
        raise Exception("Error in the pairwise preferences file: the text file is supposed to contain pairwise preferences one pair per row, i.e. the data point corresponding to the first index is preferred over the data point corresponding to the second index.\n")
    return data


def read_qids(fname):
    """Reads the query id file, used typically with ranking
    
    Parameters
    ----------
    fname : string
        input file name
        
    Returns
    -------
    Q : list of n_queries index lists
    """
    f = open(fname)
    qids = []
    for line in f:
        qid = line.strip()
        qids.append(qid)
    #Check that at least some queries contain more than one example
    if len(qids) == len(set(qids)):
        raise Exception("Error in the qid file: all the supplied queries consist only of a single example\n")
    f.close()
    Q = mapQids(qids)
    return Q

def mapQids(qids):
    q_partition = []
    prev = qids[0]
    query = [0]
    for i in range(1,len(qids)):
        if qids[i] == prev:
            query.append(i)
        else:
            q_partition.append(query)
            prev = qids[i]
            query = [i]
    q_partition.append(query)
    return q_partition


def loadtxtint(fname):
    return loadtxt(fname, dtype=int)

DEFAULT_READERS = {
                   "spmatrix": read_sparse,
                   "matrix": loadtxt,#read_dense,dtype=<type 'float'>
                   "qids": read_qids,
                   "preferences": loadtxt,#read_preferences,
                   "index_partition":read_folds, 
                   'model': cPickleload,#read_pickle,
                   'basis_vectors_variable_type': loadtxtint,#read_bvectors,
                   'data_set': read_svmlight,
                   }

