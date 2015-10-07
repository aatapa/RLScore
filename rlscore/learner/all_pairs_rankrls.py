import numpy as np
from numpy import array, eye, float64, multiply, mat, ones, sqrt, sum, zeros
import numpy.linalg as la

from rlscore.learner.abstract_learner import AbstractSvdLearner
from rlscore.utilities import array_tools
from rlscore.utilities import creators
from rlscore.measure.measure_utilities import UndefinedPerformance

class AllPairsRankRLS(AbstractSvdLearner):
    """RankRLS algorithm for learning to rank
    
    Implements the learning algorithm for learning from a single
    global ranking. For query-structured data, see LabelRankRLS.
    Uses a training algorithm that is cubic either in the
    number of training examples, or dimensionality of feature
    space (linear kernel).
    
    Computational shortcut for N-fold cross-validation: computeHO
    
    Computational shortcut for leave-pair-out: computePairwiseCV
    
    Computational shortcut for parameter selection: solve
    
    There are three ways to supply the training data for the learner.
    
    1. X: supply the data matrix directly, by default
    RLS will use the linear kernel.
    
    2. kernel_obj: supply the kernel object that has been initialized
    using the training data.
    
    3. kernel_matrix: supply user created kernel matrix, in this setting RLS
    is unable to return the model, but you may compute cross-validation
    estimates or access the learned parameters from the variable self.A

    Parameters
    ----------
    Y: {array-like}, shape = [n_samples] or [n_samples, n_labels]
        Training set labels
    regparam: float (regparam > 0)
        regularization parameter
    X: {array-like, sparse matrix}, shape = [n_samples, n_features], optional
        Data matrix
    kernel_obj: kernel object, optional
        kernel object, initialized with the training set
    kernel_matrix: : {array-like}, shape = [n_samples, n_samples], optional
        kernel matrix of the training set
        
    References
    ----------
    RankRLS algorithm and the leave-pair-out cross-validation method implemented in
    the method 'computePairwiseCV' are described in [1]_. For an experimental evaluation
    on using leave-pair-out for AUC-estimation, see [2]_.
    
    .. [1] Tapio Pahikkala, Evgeni Tsivtsivadze, Antti Airola, Jouni Jarvinen, and Jorma Boberg.
    An efficient algorithm for learning to rank from preference graphs.
    Machine Learning, 75(1):129-165, 2009.
    
    .. [2] Antti Airola, Tapio Pahikkala, Willem Waegeman, Bernard De Baets, Tapio Salakoski.
    An Experimental Comparison of Cross-Validation Techniques for Estimating the Area Under the ROC Curve.
    Computational Statistics & Data Analysis 55(4), 1828-1844, 2011.

    """
    
    #def __init__(self, svdad, Y, regparam=1.0):
    def __init__(self, **kwargs):
        self.svdad = creators.createSVDAdapter(**kwargs)
        self.Y = array_tools.as_labelmatrix(kwargs["Y"])
        if kwargs.has_key("regparam"):
            self.regparam = float(kwargs["regparam"])
        else:
            self.regparam = 1.
        self.svals = self.svdad.svals
        self.svecs = self.svdad.rsvecs
        self.size = self.Y.shape[0]

    def createLearner(cls, **kwargs):
        #new_kwargs = {}
        #new_kwargs["svdad"] = creators.createSVDAdapter(**kwargs)
        #new_kwargs["Y"] = kwargs["Y"]
        #if kwargs.has_key("regparam"):
        #    new_kwargs['regparam'] = kwargs["regparam"]
        #learner = cls(**new_kwargs)
        learner = cls(**kwargs)
        return learner
    createLearner = classmethod(createLearner)

    def train(self):
        self.solve()    
    
    def solve(self, regparam=1.0):
        """Trains the learning algorithm, using the given regularization parameter.
           
        Parameters
        ----------
        regparam: float (regparam > 0)
            regularization parameter
        """ 
        if not hasattr(self, "multiplyright"):
            
            #Eigenvalues of the kernel matrix
            self.evals = multiply(self.svals, self.svals)
            
            #Temporary variables
            ssvecs = multiply(self.svecs, self.svals)
            J = mat(ones((self.size, 1), dtype=float64))
            
            #These are cached for later use in solve function
            self.lowrankchange = ssvecs.T * J[range(ssvecs.shape[0])]
            self.multipleright = ssvecs.T * (self.size * self.Y - J * (J.T * self.Y))
        
        self.regparam = regparam
        
        #Compute the eigenvalues determined by the given regularization parameter
        neweigvals = 1. / (self.size * self.evals + regparam)
        
        P = self.lowrankchange
        nP = multiply(neweigvals.T, P)
        fastinv = 1. / (-1. + P.T * nP)
        self.A = self.svecs * multiply(1. / self.svals.T, \
            (multiply(neweigvals.T, self.multipleright) \
            - nP * (fastinv * (nP.T * self.multipleright))))
    
    
    def computePairwiseCV(self, pairs, oind=0):
        """Method for computing leave-pair-out predictions for a trained RankRLS.
        
        Parameters
        ----------
        pairs: list of index pairs, length = [n_preferences]
            list of index pairs for which the leave-pair-out predictions are calculated
        oind: int
            index of the label for which the leave-pair-out predictions should be computed
        
        Returns
        -------
        results : list of float pairs
            leave-pair-out predictions
        """
        
        evals, svecs = self.evals, self.svecs
        m = self.size
        
        Y = self.Y
        
        #This is, in the worst case, a cubic operation.
        #If there are multiple outputs,
        #this operation should be common for them all. THIS IS TO BE FIXED!
        def computeG():
            regparam = self.regparam
            G = svecs * multiply(multiply(evals, 1. / ((m - 2.) * evals + regparam)).T, svecs.T)
            return G
        G = computeG()
        
        GDY = (self.size - 2.) * G * Y
        GC = sum(G, axis=1)
        
        CTGC = sum(GC)

        
        CTY = sum(Y, axis=0)[0, oind]
        CTGDY = sum(GDY, axis=0)[0, oind]
        
        sm2 = self.size - 2.
        sqrtsm2 = sqrt(sm2)
        
        #Array is faster to access than matrix
        G = array(G)
        
        #Lists are faster to access than matrices or arrays
        def hack():
            GDY_ = []
            sqrtsm2GDY_ = []
            GC_ = []
            Y_ = []
            BTY_ = []
            Gdiag_ = []
            sm2Gdiag_ = []
            BTGBBTY_ = []
            for i in range(m):
                GDYi = GDY[i, oind]
                GDY_.append(GDYi)
                sqrtsm2GDY_.append(sqrtsm2 * GDYi)
                GC_.append(GC[i, 0])
                Yi = Y[i, oind]
                Y_.append(Yi)
                BTY_.append(sqrtsm2 * Yi)
                Gii = G[i, i]
                Gdiag_.append(Gii)
                sm2Gdiag_.append(sm2 * Gii - 1.)
                BTGBBTY_.append(sm2 * Gii * sqrtsm2 * Yi)
            return GDY_, sqrtsm2GDY_, GC_, Y_, BTY_, Gdiag_, sm2Gdiag_, BTGBBTY_
        GDY_, sqrtsm2GDY_, GC_, Y_, BTY_, Gdiag_, sm2Gdiag_, BTGBBTY_ = hack()
        
        results = []
        
        #This loops through the list of hold-out pairs.
        #Each pair is handled in a constant time.
        def looppairs(results):
            for i, j in pairs:
                
                Gii = Gdiag_[i]
                Gij = G[i, j]
                Gjj = Gdiag_[j]
                
                GCi = GC_[i]
                GCj = GC_[j]
                
                Yi = Y_[i]
                Yj = Y_[j]
                
                GDYi = GDY_[i]
                GDYj = GDY_[j]
                
                BTY0 = CTY - Yi - Yj
                BTY1 = BTY_[i]
                BTY2 = BTY_[j]
                
                GiipGij = Gii + Gij
                GijpGjj = Gij + Gjj
                GCipGCj = GCi + GCj
                
                BTGB00 = GiipGij + GijpGjj + CTGC - GCipGCj - GCipGCj
                BTGB01 = sqrtsm2 * (GCi - GiipGij)
                BTGB02 = sqrtsm2 * (GCj - GijpGjj)
                BTGB12 = sm2 * Gij
                
                BTGLY0 = CTGDY - (GDYi + GDYj + BTGB00 * BTY0 + BTGB01 * BTY1 + BTGB02 * BTY2)
                BTGLY1 = sqrtsm2GDY_[i] - (BTGB01 * BTY0 + BTGBBTY_[i] + BTGB12 * BTY2)
                BTGLY2 = sqrtsm2GDY_[j] - (BTGB02 * BTY0 + BTGB12 * BTY1 + BTGBBTY_[j])
                
                BTGB00m1 = BTGB00 - 1.
                BTGB11m1 = sm2Gdiag_[i]
                BTGB22m1 = sm2Gdiag_[j]
                
                CF00 = BTGB11m1 * BTGB22m1 - BTGB12 * BTGB12
                CF01 = -BTGB01 * BTGB22m1 + BTGB12 * BTGB02
                CF02 = BTGB01 * BTGB12 - BTGB11m1 * BTGB02
                CF11 = BTGB00m1 * BTGB22m1 - BTGB02 * BTGB02
                CF12 = -BTGB00m1 * BTGB12 + BTGB01 * BTGB02
                CF22 = BTGB00m1 * BTGB11m1 - BTGB01 * BTGB01
                
                invdeter = 1. / (BTGB00m1 * CF00 + BTGB01 * CF01 + BTGB02 * CF02)
                
                b0 = invdeter * (CF00 * BTGLY0 + CF01 * BTGLY1 + CF02 * BTGLY2) + BTY0
                b1 = invdeter * (CF01 * BTGLY0 + CF11 * BTGLY1 + CF12 * BTGLY2) + BTY1
                b2 = invdeter * (CF02 * BTGLY0 + CF12 * BTGLY1 + CF22 * BTGLY2) + BTY2
                
                t1 = -b0 + sqrtsm2 * b1
                t2 = -b0 + sqrtsm2 * b2
                F0 = GDYi - (Gii * t1 + Gij * t2 + GCi * b0)
                F1 = GDYj - (Gij * t1 + Gjj * t2 + GCj * b0)
                
                results.append((F0, F1))
        looppairs(results)
        #allresults.append(results)
        # if Y.shape[1] > 1:
            # allresultsvec = []
            # for pairind in range(len(pairs)):
                # F0 = mat(zeros((1, Y.shape[1]), dtype = float64))
                # F1 = mat(zeros((1, Y.shape[1]), dtype = float64))
                # for oind in range(Y.shape[1]):
                    # F0[0, oind] = allresults[oind][pairind][0]
                    # F1[0, oind] = allresults[oind][pairind][1]
                # allresultsvec.append((F0,F1))
            # return allresultsvec
        # else:
            # return allresults[0]
        return results
    
    
    def computeHO(self, indices):
        """Computes hold-out predictions for a trained RankRLS.
        
        Parameters
        ----------
        indices: list of indices, shape = [n_hsamples]
            list of indices of training examples belonging to the set for which the hold-out predictions are calculated. The list can not be empty.

        Returns
        -------
        F : matrix, shape = [n_hsamples, n_labels]
            holdout predictions
        """
        if len(indices) == 0:
            raise Exception('Hold-out predictions can not be computed for an empty hold-out set.')
        
        if len(indices) != len(set(indices)):
            raise Exception('Hold-out can have each index only once.')
        
        Y = self.Y
        m = self.size
        
        evals, V = self.evals, self.svecs
        
        #results = []
        
        C = mat(zeros((self.size, 3), dtype=float64))
        onevec = mat(ones((self.size, 1), dtype=float64))
        for i in range(self.size):
            C[i, 0] = 1.
        
        
        VTY = V.T * Y
        VTC = V.T * onevec
        CTY = onevec.T * Y
        
        holen = len(indices)
        
        newevals = multiply(evals, 1. / ((m - holen) * evals + self.regparam))
        
        R = mat(zeros((holen, holen + 1), dtype=float64))
        for i in range(len(indices)):
            R[i, 0] = -1.
            R[i, i + 1] = sqrt(self.size - float(holen))
        
        Vho = V[indices]
        Vhov = multiply(Vho, newevals)
        Ghoho = Vhov * Vho.T
        GCho = Vhov * VTC
        GBho = Ghoho * R
        for i in range(len(indices)):
            GBho[i, 0] += GCho[i, 0]
        
        CTGC = multiply(VTC.T, newevals) * VTC
        RTGCho = R.T * GCho
        
        BTGB = R.T * Ghoho * R
        for i in range(len(indices) + 1):
            BTGB[i, 0] += RTGCho[i, 0]
            BTGB[0, i] += RTGCho[i, 0]
        BTGB[0, 0] += CTGC[0, 0]
        
        BTY = R.T * Y[indices]
        #BTY[0, 0] += CTY[0, 0]
        BTY[0] = BTY[0] + CTY[0]
        
        GDYho = Vhov * (self.size - holen) * VTY
        GLYho = GDYho - GBho * BTY
        
        CTGDY = multiply(VTC.T, newevals) * (self.size - holen) * VTY
        BTGLY = R.T * GDYho - BTGB * BTY
        #BTGLY[0, 0] += CTGDY[0, 0]
        BTGLY[0] = BTGLY[0] + CTGDY[0]
        
        F = GLYho - GBho * la.inv(-mat(eye(holen + 1)) + BTGB) * BTGLY
        
        #results.append(F)
        #return results
        return F
        
    def computeLOO(self):
        
        LOO = mat(zeros((self.size, self.ysize), dtype=float64))
        for i in range(self.size):
            LOO[i,:] = self.computeHO([i])
        return LOO
    
    def reference(self, pairs):
        
        evals, evecs = self.evals, self.svecs
        Y = self.Y
        m = self.size
        
        
        results = []
        
        D = mat(zeros((self.size, 1), dtype=float64))
        C = mat(zeros((self.size, 3), dtype=float64))
        for i in range(self.size):
            D[i, 0] = self.size - 2.
            C[i, 0] = 1.
        lamb = self.regparam
        
        G = evecs * multiply(multiply(evals, 1. / ((m - 2.) * evals + lamb)).T, evecs.T)
        
        
        DY = multiply(D, Y)
        GDY = G * DY
        GC = G * C
        CTG = GC.T
        CT = C.T
        CTGC = CT * GC
        CTY = CT * Y
        #GCCTY = (G * C) * (C.T * Y)
        CTGDY = CT * GDY
        
        minusI3 = -mat(eye(3))
        for i, j in pairs:
            hoinds = [i, j]
            
            R = mat(zeros((2, 3), dtype=float64))
            R[0, 0] = -1.
            R[1, 0] = -1.
            R[0, 1] = sqrt(self.size - 2.)
            R[1, 2] = sqrt(self.size - 2.)
            #RT = R.T
            
            GBho = GC[hoinds] + G[np.ix_(hoinds, hoinds)] * R
            
            BTGB = CTGC \
                + R.T * GC[hoinds] \
                + CTG[:, hoinds] * R \
                + R.T * G[np.ix_(hoinds, hoinds)] * R
            
            BTY = CTY + R.T * Y[hoinds]
            
            GLYho = GDY[hoinds] - GBho * BTY
            BTGLY = CTGDY + R.T * GDY[hoinds] - BTGB * BTY
            
            F = GLYho - GBho * la.inv(minusI3 + BTGB) * BTGLY
            
            results.append(F)
        return results

class NfoldCV(object):
    
    def __init__(self, learner, measure, folds):
        self.rls = learner
        self.measure = measure
        self.folds = folds
        
    def cv(self, regparam):
        rls = self.rls
        folds = self.folds
        measure = self.measure
        rls.solve(regparam)
        Y = rls.Y
        performances = []
        for fold in folds:
            P = rls.computeHO(fold)
            try:
                performance = measure(Y[fold], P)
                performances.append(performance)
            except UndefinedPerformance:
                pass
            #performance = measure_utilities.aggregate(performances)
        if len(performances) > 0:
            performance = np.mean(performances)
        else:
            raise UndefinedPerformance("Performance undefined for all folds")
        return performance

class LPOCV(object):
    
    
    def __init__(self, learner, measure):
        self.rls = learner
        self.measure = measure

    def cv(self, regparam):
        rls = self.rls
        rls.solve(regparam)
        Y = rls.Y
        perfs = []
        #special handling for concordance index / auc
        if self.measure.func_name in ["cindex", "auc"]:
            for index in range(Y.shape[1]):
                pairs = []
                for i in range(Y.shape[0] - 1):
                    for j in range(i + 1, Y.shape[0]):
                        if Y[i, index] > Y[j, index]:
                            pairs.append((i, j))
                        elif Y[i, index] < Y[j, index]:
                            pairs.append((j, i))
                if len(pairs) > 0:
                    pred = rls.computePairwiseCV(pairs, index)
                    auc = 0.
                    for pair in pred:
                        if pair[0] > pair[1]:
                            auc += 1.
                        elif pair[0] == pair[1]:
                            auc += 0.5
                    auc /= len(pred)
                    perfs.append(auc)
            if len(perfs) > 0:
                performance = np.mean(perfs)
            else:
                raise UndefinedPerformance("Performance undefined for all folds")
            return performance
        else:
            #Horribly inefficient, but maybe OK for small data sets
            pairs = []
            for i in range(Y.shape[0]):
                for j in range(Y.shape[0]):
                    pairs.append((i,j))
            for index in range(Y.shape[1]):
                preds = rls.computePairwiseCV(pairs, index)
                for i in range(len(pairs)):
                    pair = pairs[i]
                    pred = preds[i]
                    perfs.append(self.measure(np.array([Y[pair[0],index],Y[pair[1],index]]), np.array(pred)))
            perf = np.mean(perfs)
            return perf

                
        
        
    

