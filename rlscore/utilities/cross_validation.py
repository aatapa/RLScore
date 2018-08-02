#
# The MIT License (MIT)
#
# This file is part of RLScore 
#
# Copyright (c) 2015 - 2016 Tapio Pahikkala, Antti Airola
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

import random
import numpy as np

def random_folds(ssize, k, seed = None):
    #returns indices from 0...ssize-1 split to k random folds
    if seed is not None:
        myrandom = random.Random(seed)
    else:
        myrandom = random
    folds = []
    indices = set(range(ssize))
    foldsize = ssize // k
    leftover = ssize % k
    for i in range(k):
        sample_size = foldsize
        if leftover > 0:
            sample_size += 1
            leftover -= 1
        fold = myrandom.sample(indices, sample_size)
        indices = indices.difference(fold)
        folds.append(fold)
    return folds

def grid_search(crossvalidator, grid):
    #used internally for fast regularization parameter selection
    performances = []
    predictions = []
    for regparam in grid:
        perf, P = crossvalidator.cv(regparam)
        performances.append(perf)
        predictions.append(P)
    if crossvalidator.measure.iserror:
        bestparam = grid[np.argmin(performances)]
    else:
        bestparam = grid[np.argmax(performances)]
    learner = crossvalidator.rls
    learner.solve(bestparam)
    return np.array(performances), predictions, bestparam

def map_ids(ids):
    q_partition = []
    prev = ids[0]
    query = [0]
    for i in range(1,len(ids)):
        if ids[i] == prev:
            query.append(i)
        else:
            q_partition.append(query)
            prev = ids[i]
            query = [i]
    q_partition.append(query)
    return q_partition
