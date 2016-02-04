import random
import numpy as np

def random_folds(ssize, k, seed = None):
    #returns indices from 0...ssize-1 split to k random folds
    if seed != None:
        myrandom = random.Random(seed)
    else:
        myrandom = random
    folds = []
    indices = set(range(ssize))
    foldsize = ssize / k
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
