import numpy as np

def grid_search(crossvalidator, grid):
    performances = []
    for regparam in grid:
        perf = crossvalidator.cv(regparam)
        performances.append(perf)
    if crossvalidator.measure.iserror:
        bestparam = grid[np.argmin(performances)]
    else:
        bestparam = grid[np.argmax(performances)]
    learner = crossvalidator.rls
    learner.solve(bestparam)        
    return learner, performances
