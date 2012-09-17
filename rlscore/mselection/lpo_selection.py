import numpy as np

from rlscore.mselection.abstract_selection import AbstractSelection
from rlscore.learner import AllPairsRankRLS
from rlscore.measure import measure_utilities

class LPOSelection(AbstractSelection):
    """leave-pair-out cross-validation for model selection"""
    
    
    def estimatePerformance(self, learner):
        """Leave-pair-out estimate of performance
        
        @param learner: trained learner object
        @type learner: RLS
        @return: estimated performance for the learner
        @rtype: float"""
        if not isinstance(self.learner, AllPairsRankRLS):
            raise Exception("LPO cross-validation implemented only for RankRLS")
        predictions = []
        performances = []
        for i in range(self.Y.shape[1]):
            #pairs = self.measure.getPairs(self.Y, i)
            pairs = getPairs(self.Y, i)
            pred = learner.computePairwiseCV(pairs, i)
            #perf = self.measure.pairwisePerformance(pairs, self.Y, i, pred)
            perf = pairwisePerformance(pairs, self.Y, i, pred)
            predictions.append(pred)
            performances.append(perf)
        #performance = measure_utilities.aggregate(performances)
        performance = np.mean(performances)
        return performance


def getPairs(Y, index):
    """Returns all positive-negative pairs.
    
    @param Y: matrix of correct labels, each column corresponds to one task
    @type Y: numpy matrix
    @return: list of lists of index pairs
    @rtype list of lists of integer pairs"""
    pairs = []
    tsetsize = Y.shape[0]
    for i in range(tsetsize - 1):
        for j in range(i + 1, tsetsize):
            if Y[i, index] > Y[j, index]:
                pairs.append((i, j))
            elif Y[i, index] < Y[j, index]:
                pairs.append((j, i))
    return pairs


def pairwisePerformance(pairs, Y, index, predicted):
    """Used for LPO-cross-validation. Supplied pairs should consist of all positive-negative
    pairs.
    
    @param pairs: a list of tuples of length two, containing the indices of the pairs in Y
    @type pairs: list of integer pairs
    @param Y: matrix of correct labels, each column corresponds to one task
    @type Y: numpy matrix
    @param index: the index of the task considered, this corresponding to a given column of Y
    @type index: integer
    @param predicted: a list of tuples of length two, containing the predictions for the pairs
    @type predicted: list of float pairs
    @return: performance
    @rtype: float"""
    assert len(pairs) == len(predicted)
    if len(pairs) == 0:
        return None
    auc = 0.
    for pair in predicted:
        if pair[0] > pair[1]:
            auc += 1.
        elif pair[0] == pair[1]:
            auc += 0.5
    auc /= len(predicted)
    return auc

