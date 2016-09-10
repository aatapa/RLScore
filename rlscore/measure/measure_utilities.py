#
# The MIT License (MIT)
#
# This file is part of RLScore 
#
# Copyright (c) 2010 - 2016 Tapio Pahikkala, Antti Airola
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

import numpy as np

def wrapper(measure, Y, Y_predicted, qids):
    Y = np.mat(Y)
    Y_predicted = np.mat(Y_predicted)
    qids_perfs = []
    for inds in qids:
        Y_sub = Y[inds]
        Y_predicted_sub = Y_predicted[inds]
        perfs = measure.getPerformance(Y_sub, Y_predicted_sub)
        qids_perfs.append(perfs)
    #quite a bit juggling follows to handle the fact, that nans encode
    #queries for which performance is undefined (happens sometimes
    #in ranking
    #
    #count the number of non-nan values in each column
    perfs = np.vstack(qids_perfs)
    normalizers = np.isnan(perfs)
    normalizers = np.logical_not(normalizers)
    normalizers = np.sum(normalizers, axis=0)
    normalizers = np.where(normalizers>0,normalizers,np.nan)
    #turns nans into zeroes
    perfs = np.nan_to_num(perfs)
    perfs = np.sum(perfs, axis=0)
    perfs = perfs/normalizers
    return perfs

def qids_to_splits(qids):
    """Sets the qid parameters of the training examples. The list must have as many qids as there are training examples.
    
    @param qids: A list of qid parameters.
    @type qids: List of integers."""
    qidmap = {}
    i = 0
    for qid in qids:
        if not qid in qidmap:
            qidmap[qid] = i
            i+=1
    new_qids = []
    for qid in qids:
        new_qids.append(qidmap[qid])
    qidcount = np.max(new_qids)+1
    splits = [[] for i in range(qidcount)]
    for i, qid in enumerate(new_qids):
        splits[qid].append(i) 
    return splits


def aggregate(performances):
    normalizer = np.sum(np.logical_not(np.isnan(performances)))
    if normalizer == 0:
        return np.nan
    else:
        performances = np.nan_to_num(performances)
        return np.sum(performances)/normalizer
    
def multitask(Y, P, f):
    perfs = []
    for i in range(Y.shape[1]):
        perfs.append(f(Y[:,i], P[:,i]))
    return perfs
    
class UndefinedPerformance(Exception):
    """Used to indicate that the performance is not defined for the
    given predictions and outputs."""
    #Examples of this type of issue are disagreement error, which
    #is undefined when all the true labels are the same, and
    #recall, which is not defined if there are no relevant
    #instances in the data set.

    def __init__(self, value):
        """Initialization
        
        @param value: the error message
        @type value: string"""
        self.value = value

    def __str__(self):
        return repr(self.value)
