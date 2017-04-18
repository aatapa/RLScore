#
# The MIT License (MIT)
#
# This file is part of RLScore 
#
# Copyright (c) 2008 - 2016 Tapio Pahikkala, Antti Airola
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from scipy import sparse
from numpy import float64, loadtxt
import numpy as np

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



def read_sparse(fname, fdim=None):
    """Reads in a sparse n x m matrix from a file with n rows.
    
    Format is of the type 0:1.5 3:4.2 7:1.1 ...
    with each line containing index:value pairs with indices
    ranging from 0...n_features-1, and only indices with non-zero values
    being present in the file.
    
    Parameters
    ----------
    fname : string
        input file name
    fdim: int
        number of dimensions, if None estimated from data file
        
    Returns
    -------
    X : sparse matrix (csr)
    """
    #each row represents an instance, each column a feature
    f = open(fname)
    rows = []
    columns = []
    values = []
    linecounter = 0
    for line in f:
        linecounter += 1
        #Empty lines and commented lines are passed over
        if len(line.strip()) == 0 or line[0] == '#':
            print("Warning: no inputs on line %d" % linecounter)
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
                if value != 0. and (fdim is None or index < fdim): 
                    columns.append(index)
                    rows.append(linecounter-1)
                    values.append(value)
            except ValueError:
                raise Exception("Error when reading in feature file: feature:value pair %s on line %d is not well-formed\n" % (att_val, linecounter))
            if not index > previous:
                raise Exception("Error when reading in feature file: line %d features must be in ascending order\n" % (linecounter))
            previous = index
    #That's all folks
    if fdim is None:
        X = sparse.coo_matrix((values,(rows,columns)), dtype=float64)
    else:
        rdim = np.max(rows)+1
        X = sparse.coo_matrix((values,(rows,columns)), (rdim, fdim), dtype=float64)
    X = X.tocsr()
    f.close()
    return X


def read_svmlight(fname, fdim=None):
    """ Loads examples from an SVM-light format data file. The
    file contains attributes, one label per example and optionally qids.
    
    Parameters
    ----------
    fname : string
        input file name
    fdim: int
        number of dimensions, if None estimated from data file
        
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
            if qids is None:
                if linecounter > 1:
                    raise Exception("Error when reading in SVMLight file: Line %d has a qid, previous lines did not have qids defined" % (linenumber))   
                else:
                    qids = [qid]
            else:
                qids.append(qid)
        else:
            if qids is not None:
                raise Exception("Error when reading in SVMLight file: Line %d has no qid, previous lines had qids defined" % (linenumber))
        attributes = line
        #Multiple labels are allowed, but each instance must have the
        #same amount of them. Labels must be real numbers.
        labels = labels.split("|")
        if labelcount is None:
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
                if value != 0. and (fdim is None or index < fdim): 
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
    if fdim is not None:
        feaspace_dim = fdim
    X = sparse.coo_matrix((values,(rows,columns)),(linecounter, feaspace_dim), dtype=float64)
    X = X.tocsr()
    Y = np.array(all_outputs)
    return X, Y, qids


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
    return qids

def loadtxtint(fname):
    return loadtxt(fname, dtype=int)

