
from random import *

from numpy import *

from rlscore.learner.abstract_learner import AbstractSvdLearner
from rlscore.learner.abstract_learner import AbstractIterativeLearner
from rlscore.utilities import array_tools
from rlscore.utilities import creators

class MMC(AbstractSvdLearner, AbstractIterativeLearner):
    """RLS-based maximum-margin clustering.
    
    Performs stochastic search, that aims to find a labeling of the data such
    that would minimize the RLS-objective function.
    
    There are three ways to supply the training data for the learner.
    
    1. train_features: supply the data matrix directly, by default
    MMC will use the linear kernel.
    
    2. kernel_obj: supply the kernel object that has been initialized
    using the training data.
    
    3. kernel_matrix: supply user created kernel matrix.

    Parameters
    ----------
    train_features: {array-like, sparse matrix}, shape = [n_samples, n_features]
        Data matrix
    regparam: float (regparam > 0)
        regularization parameter
    number_of_clusters: int (number_of_clusters >1)
        number of clusters to be found
    train_features: {array-like, sparse matrix}, shape = [n_samples, n_features], optional
        Data matrix
    kernel_obj: kernel object, optional
        kernel object, initialized with the training set
    kernel_matrix: : {array-like}, shape = [n_samples, n_samples], optional
        kernel matrix of the training set
        
    References
    ----------
    
    The MMC algorithm is described in [1]_.
           
    ..[1] Fabian Gieseke, Tapio Pahikkala,  and Oliver Kramer.
    Fast Evolutionary Maximum Margin Clustering.
    Proceedings of the 26th International Conference on Machine Learning,
    361-368, ACM, 2009.
    """
    
    #def __init__(self, svdad, number_of_clusters=2, regparam=1.0, train_labels = None, fixed_indices=None, callback=None):
    def __init__(self, **kwargs):
        self.svdad = creators.createSVDAdapter(**kwargs)
        self.svals = self.svdad.svals
        self.svecs = self.svdad.rsvecs
        self.regparam = float(kwargs["regparam"])
        self.constraint = 0
        #if not kwargs.has_key('number_of_clusters'):
        #    raise Exception("Parameter 'number_of_clusters' must be given.")
        if kwargs.has_key("number_of_clusters"):
            self.labelcount = int(kwargs["number_of_clusters"])
        else:
            self.labelcount = 2
        if self.labelcount == 2:
            self.oneclass = True
        else:
            self.oneclass = False
        if kwargs.has_key("callback"):
            self.callbackfun = kwargs["callback"]
        else:
            self.callbackfun = None
        if kwargs.has_key("train_labels"):
            train_labels = kwargs["train_labels"]
        else:
            train_labels = None
        if train_labels != None:
            Y_orig = array_tools.as_labelmatrix(train_labels)
            if Y_orig.shape[1] == 1:
                self.Y = zeros((Y_orig.shape[0], 2))
                self.Y[:, 0] = Y_orig
                self.Y[:, 1] = - Y_orig
                self.oneclass = True
            else:
                self.Y = Y_orig.copy()
                self.oneclass = False
            for i in range(self.Y.shape[0]):
                largestind = 0
                largestval = self.Y[i, 0]
                for j in range(self.Y.shape[1]):
                    if self.Y[i, j] > largestval:
                        largestind = j
                        largestval = self.Y[i, j]
                    self.Y[i, j] = -1.
                self.Y[i, largestind] = 1.
        else:
            size = self.svecs.shape[0]
            ysize = self.labelcount
            if self.labelcount == None: self.labelcount = 2
            self.Y = RandomLabelSource(size, ysize).readLabels()
        self.size = self.Y.shape[0]
        self.labelcount = self.Y.shape[1]
        self.classvec = - ones((self.size), dtype = int32)
        self.classcounts = zeros((self.labelcount), dtype = int32)
        for i in range(self.size):
            clazzind = 0
            largestlabel = self.Y[i, 0]
            for j in range(self.labelcount):
                if self.Y[i, j] > largestlabel:
                    largestlabel = self.Y[i, j]
                    clazzind = j
            self.classvec[i] = clazzind
            self.classcounts[clazzind] = self.classcounts[clazzind] + 1
        
        self.svecs_list = []
        for i in range(self.size):
            self.svecs_list.append(self.svecs[i].T)
        self.fixedindices = []
        if kwargs.has_key("fixed_indices"):
            self.fixedindices = kwargs["fixed_indices"]
        else:
            self.fixedindices = []
        self.results = {}
    
    
    def createLearner(cls, **kwargs):
        learner = cls(**kwargs)
        return learner
    createLearner = classmethod(createLearner)
    
    
    def train(self):
        """Trains the learning algorithm.
        """
        self.solve(self.regparam)
    
    
    def solve(self, regparam):
        """Trains the learning algorithm, using the given regularization parameter.
               
        Parameters
        ----------
        regparam: float (regparam > 0)
            regularization parameter
        """
        self.regparam = regparam
        
        #Some counters for bookkeeping
        self.stepcounter = 0
        self.flipcounter = 0
        self.nochangecounter = 0
        
        '''
        #Cached results
        self.evals = multiply(self.svals, self.svals)
        self.newevals = 1. / (self.evals + regparam)
        newevalslamtilde = multiply(self.evals, self.newevals)
        A1 = multiply(newevalslamtilde, newevalslamtilde)
        A2 = - 2 * newevalslamtilde
        A3 = self.regparam * multiply(newevalslamtilde, self.newevals)
        self.D = A1 + A2 + A3
        '''
        
        #Cached results
        self.evals = multiply(self.svals, self.svals)
        self.newevals = 1. / (self.evals + self.regparam)
        newevalslamtilde = multiply(self.evals, self.newevals)
        self.D = sqrt(newevalslamtilde)
        #self.D = -newevalslamtilde
        
        self.VTY = self.svecs.T * self.Y
        DVTY = multiply(self.D.T, self.svecs.T * self.Y)
        
        #Using lists in order to avoid unnecessary matrix slicings
        self.DVTY_list = []
        self.YTVDDVTY_list = []
        self.classFitnessList = []
        for i in range(self.labelcount):
            DVTY_i = DVTY[:,i]
            self.DVTY_list.append(DVTY_i)
            YTVDDVTY_i = DVTY_i.T * DVTY_i
            self.YTVDDVTY_list.append(YTVDDVTY_i)
            fitness_i = self.size - DVTY_i.T * DVTY_i
            self.classFitnessList.append(fitness_i)
        
        self.Dsvecs_list = []
        self.svecsDDsvecs_list = []
        for i in range(self.size):
            Dsvec = multiply(self.D.T, self.svecs[i].T)
            self.Dsvecs_list.append(Dsvec)
            self.svecsDDsvecs_list.append(Dsvec.T*Dsvec)
        
        self.updateA()
        
        
        converged = False
        print self.classcounts.T
        self.callback()
        while True:
            
            converged = self.roundRobin()
            print self.classcounts.T
            self.callback()
            if converged: break
        
        #for ii in range(5):
        #    self.giveAndTakeALT(1000/ (ii+1))
        if self.oneclass:
            self.Y = self.Y[:, 0]
        self.results['predicted_clusters_for_training_data'] = self.Y
        self.results['model'] = self.getModel()
    
    
    def computeGlobalFitness(self):
        fitness = 0.
        for classind in range(self.labelcount):
            fitness += self.classFitnessList[classind]
        return fitness
    
    
    def computeFlipFitnessForSingleClass(self, flipindex, classind):
        
        currentclassind = self.classvec[flipindex]
        coef = -2
        if currentclassind != classind:
            coef = 2
        
        #DVTY_new = DVTY_old + coef * self.Dsvecs_list[flipindex]
        #self.size - DVTY_new.T * DVTY_new
        
        #self.size - (DVTY_old + coef * self.Dsvecs_list[flipindex]).T * (DVTY_old + coef * self.Dsvecs_list[flipindex])
        #self.size - (DVTY_old.T*DVTY_old + 2 * coef * DVTY_old.T * self.Dsvecs_list[flipindex] + 4 * self.Dsvecs_list[flipindex].T * self.Dsvecs_list[flipindex])
        
        DVTY_old = self.DVTY_list[classind]
        #DVTY_new = DVTY_old + coef * self.Dsvecs_list[flipindex]
        fitness_old = self.classFitnessList[classind]
        #fitness_new = self.size - DVTY_new.T * DVTY_new
        fitness_new = self.size - (self.YTVDDVTY_list[classind] + 2 * coef * DVTY_old.T * self.Dsvecs_list[flipindex] + 4 * self.svecsDDsvecs_list[flipindex])
        #print fitness_new
        #print self.size - (DVTY_old.T*DVTY_old + 2 * coef * DVTY_old.T * self.Dsvecs_list[flipindex] + 4 * self.Dsvecs_list[flipindex].T * self.Dsvecs_list[flipindex])
        #print self.size - (self.YTVDDVTY_list[classind] + 2 * coef * DVTY_old.T * self.Dsvecs_list[flipindex] + 4 * self.svecsDDsvecs_list[flipindex])
        #sys.exit()
        fitnessdiff = fitness_new - fitness_old
        return fitnessdiff

    
    def flipClass(self, flipindex, newclassind, DVTY_new_currentclass=None, DVTY_new_newclass=None):
        currentclassind = self.classvec[flipindex]
        self.Y[flipindex, currentclassind] = -1.
        self.Y[flipindex, newclassind] = 1.
        self.classcounts[currentclassind] -= 1
        self.classcounts[newclassind] += 1
        self.classvec[flipindex] = newclassind
        
        if DVTY_new_currentclass == None:
            DVTY_new_currentclass = self.DVTY_list[currentclassind] - 2 * self.Dsvecs_list[flipindex]
        if DVTY_new_newclass == None:
            DVTY_new_newclass = self.DVTY_list[newclassind] + 2 * self.Dsvecs_list[flipindex]
        
        self.DVTY_list[currentclassind] = DVTY_new_currentclass
        self.DVTY_list[newclassind] = DVTY_new_newclass
    
    
    def step(self, flipindex):
        """Perform one iteration step by checking whether flipping the label indexed by flipindex improves the current state. If it does, then the label is flipped. Otherwise, the the current label vector is left unchanged.
        flipindex:    the index of the label to be flipped.
        returns:      True, if the label is flipped and False otherwise."""
        y = self.Y[flipindex, 0]
        currentclassind = self.classvec[flipindex]
        
        DVTY_old_currentclass = self.DVTY_list[currentclassind]
        fitness_old_currentclass = self.classFitnessList[currentclassind]
        DVTY_new_currentclass = DVTY_old_currentclass - 2 * self.Dsvecs_list[flipindex]
        fitness_new_currentclass = self.size - DVTY_new_currentclass.T * DVTY_new_currentclass
        fitnessdiff_currentclass = fitness_new_currentclass - fitness_old_currentclass
        
        changed = False
        
        bestclassind = None
        VTY_new_bestclass = None
        bestfitnessdiff = float('Inf')
        fitness_new_bestclass = None
        
        for newclassind in range(self.labelcount):
            
            if newclassind == currentclassind: continue
            
            DVTY_old_newclass = self.DVTY_list[newclassind]
            DVTY_new_newclass = DVTY_old_newclass + 2 * self.Dsvecs_list[flipindex]
            fitness_old_newclass = self.classFitnessList[newclassind]
            fitness_new_newclass = self.size - DVTY_new_newclass.T * DVTY_new_newclass
            fitnessdiff_newclass = fitness_new_newclass - fitness_old_newclass
            
            fitnessdiff = fitnessdiff_currentclass + fitnessdiff_newclass
            
            if fitnessdiff < 0 and fitnessdiff < bestfitnessdiff:
                bestfitnessdiff = fitnessdiff[0, 0]
                bestclassind = newclassind
                DVTY_new_bestclass = DVTY_new_newclass
                fitness_new_bestclass = fitness_new_newclass
        
        #print 'BFD', bestfitnessdiff
        
        if bestclassind != None:
            if self.classcounts[currentclassind] > self.constraint:
                
                self.Y[flipindex, currentclassind] = -1.
                self.Y[flipindex, bestclassind] = 1.
                self.classcounts[currentclassind] -= 1
                self.classcounts[bestclassind] += 1
                self.classvec[flipindex] = bestclassind
                
                self.DVTY_list[currentclassind] = DVTY_new_currentclass
                self.DVTY_list[bestclassind] = DVTY_new_bestclass
                self.classFitnessList[currentclassind] = fitness_new_currentclass
                self.classFitnessList[bestclassind] = fitness_new_bestclass
                
                changed = True
        
        return changed
    
    
    def stepLOO(self, flipindex):
        """Perform one iteration step by checking whether flipping the label indexed by flipindex improves the current state. If it does, then the label is flipped. Otherwise, the the current label vector is left unchanged.
        flipindex:    the index of the label to be flipped.
        returns:      True, if the label is flipped and False otherwise."""
        y = self.Y[flipindex, 0]
        currentclassind = self.classvec[flipindex]
        
        DVTY_old_currentclass = self.DVTY_list[currentclassind]
        #fitness_old_currentclass = self.classFitnessList[currentclassind]
        DVTY_new_currentclass = DVTY_old_currentclass - 2 * self.Dsvecs_list[flipindex]
        #fitness_new_currentclass = self.size + VTY_new_currentclass.T * multiply(self.D.T, VTY_new_currentclass)
        #fitnessdiff_currentclass = fitness_new_currentclass - fitness_old_currentclass
            
        bevals = multiply(self.evals, self.newevals)
        RV = self.Dsvecs_list[flipindex]
        right = DVTY_old_currentclass - RV * (-1.)
        #print RV.shape, bevals.shape, right.shape
        RQY = RV.T * multiply(bevals.T, right)
        RQRT = RV.T * multiply(bevals.T, RV)
        result = 1. / (1. - RQRT) * RQY
        fitnessdiff_currentclass = (-1. - result) ** 2 - (1. - result) ** 2
        
        changed = False
        
        bestclassind = None
        DVTY_new_bestclass = None
        bestfitnessdiff = 10.
        fitness_new_bestclass = None
        
        for newclassind in range(self.labelcount):
            
            if newclassind == currentclassind: continue
            
            DVTY_old_newclass = self.DVTY_list[newclassind]
            DVTY_new_newclass = DVTY_old_newclass + 2 * self.Dsvecs_list[flipindex]
            
            right = DVTY_old_newclass - RV * (-1.)
            RQY = RV.T * multiply(bevals.T, right)
            RQRT = RV.T * multiply(bevals.T, RV)
            result = 1. / (1. - RQRT) * RQY
            
            fitnessdiff_newclass = (1. - result) ** 2 - (-1. - result) ** 2
            
            fitnessdiff = fitnessdiff_currentclass + fitnessdiff_newclass
            
            if fitnessdiff < 0 and fitnessdiff < bestfitnessdiff:
                bestfitnessdiff = fitnessdiff
                bestclassind = newclassind
                DVTY_new_bestclass = DVTY_new_newclass
                #fitness_new_bestclass = fitness_new_newclass
        
        if bestclassind != None:
            if self.classcounts[currentclassind] > self.constraint:
                
                self.Y[flipindex, currentclassind] = -1.
                self.Y[flipindex, bestclassind] = 1.
                self.classcounts[currentclassind] -= 1
                self.classcounts[bestclassind] += 1
                self.classvec[flipindex] = bestclassind
                
                self.DVTY_list[currentclassind] = DVTY_new_currentclass
                self.DVTY_list[bestclassind] = DVTY_new_bestclass
                #self.classFitnessList[currentclassind] = fitness_new_currentclass
                #self.classFitnessList[bestclassind] = fitness_new_bestclass
                
                changed = True
        
        return changed
    
    
    def updateA(self):
        self.A = self.svecs * multiply(self.newevals.T, self.VTY)
        #if self.U == None:
        #    self.A = self.svecs * multiply(self.newevals.T, self.VTY)
        #else:
        #    bevals = multiply(self.svals, self.newevals)
        #    self.A = self.U.T * multiply(bevals.T, self.VTY)
    
    
    def roundRobin(self, LOO = False):
        
        localstepcounter = 0
        
        converged = False
        
        flipindex = self.size - 1
        allinds = set(range(self.size)) - set(self.fixedindices)
        
        for flipindex in allinds:
            if LOO:
                flipped = self.stepLOO(flipindex)
            else:
                flipped = self.step(flipindex)
            self.stepcounter += 1
            localstepcounter += 1
            if flipped:
                self.flipcounter += 1
                self.nochangecounter = 0
            else:
                self.nochangecounter += 1
            if self.nochangecounter >= self.size:
                converged = True
                break
        
        self.updateA()
        return converged
        
    
    def giveAndTake(self, howmany):
        
        allinds = set(range(self.size))
        allinds = allinds - set(self.fixedindices)
        allinds = list(allinds)
        random.shuffle(allinds)
        
        for clazz in range(self.labelcount):
            givelist = []
            for flipindex in allinds:
                if self.classvec[flipindex] != clazz:
                    continue
                else:
                    bestfit = None
                    bestclass = None
                    ffit1 = self.computeFlipFitnessForSingleClass(flipindex, clazz)
                    for newclassind in range(self.labelcount):
                        if newclassind == clazz: continue
                        ffit2 = self.computeFlipFitnessForSingleClass(flipindex, newclassind)
                        #flipfit = self.computeFlipFitness(flipindex, newclassind)
                        flipfit = ffit1 + ffit2
                        if bestfit == None or bestfit > flipfit:
                            bestfit = flipfit
                            bestclass = newclassind
                    givelist.append((bestfit,bestclass,flipindex))
            givelist.sort()
            
            for i in range(min(howmany,len(givelist))):
                fit, newclassind, flipindex = givelist[i]
                self.flipClass(flipindex, newclassind)
        
            takelist = []
            for flipindex in allinds:
                if self.classvec[flipindex] != clazz:
                    ffit1 = self.computeFlipFitnessForSingleClass(flipindex, self.classvec[flipindex])
                    ffit2 = self.computeFlipFitnessForSingleClass(flipindex, clazz)
                    flipfit = ffit1 + ffit2
                    #fflipfit = self.computeFlipFitness(flipindex, clazz)
                    takelist.append((flipfit,flipindex))
            takelist.sort()
            
            takeind = 0
            takecount = 0
            while True:
                fit, flipindex = takelist[takeind]
                oldclazz = self.classvec[flipindex]
                if self.classcounts[oldclazz] > self.constraint: 
                    self.flipClass(flipindex, clazz)
                    takecount += 1
                    if takecount >= min(howmany, len(takelist)): break
                takeind += 1
        
        #for clazz in range(self.labelcount):
        #    print self.computeFitnessForOneClass(self.VTY[:,clazz])

        self.updateA()
    
    
    
    
    def giveAndTakeALT(self, howmany):
        
        allinds = set(range(self.size))
        allinds = allinds - set(self.fixedindices)
        allinds = list(allinds)
        random.shuffle(allinds)
        
        for clazz in range(self.labelcount):
            givelist = []
            for flipindex in allinds:
                if self.classvec[flipindex] != clazz:
                    continue
                else:
                    ffit1 = self.computeFlipFitnessForSingleClass(flipindex, clazz)
                    bestfit = None
                    bestclass = None
                    for newclassind in range(self.labelcount):
                        if newclassind == clazz: continue
                        ffit2 = self.computeFlipFitnessForSingleClass(flipindex, newclassind)
                        #flipfit = self.computeFlipFitness(flipindex, newclassind)
                        flipfit = ffit1 + ffit2
                        if bestfit == None or bestfit > flipfit:
                            bestfit = flipfit
                            bestclass = newclassind
                    givelist.append((ffit1, bestclass, flipindex))
            givelist.sort()
            
            for i in range(min(howmany,len(givelist))):
                fit, newclassind, flipindex = givelist[i]
                self.flipClass(flipindex, newclassind)
        
            takelist = []
            for flipindex in allinds:
                if self.classvec[flipindex] != clazz:
                    #ffit1 = self.computeFlipFitnessForSingleClass(flipindex, self.classvec[flipindex])
                    ffit2 = self.computeFlipFitnessForSingleClass(flipindex, clazz)
                    #flipfit = ffit1 + ffit2
                    #fflipfit = self.computeFlipFitness(flipindex, clazz)
                    takelist.append((ffit2,flipindex))
            takelist.sort()
            
            takeind = 0
            takecount = 0
            while True:
                fit, flipindex = takelist[takeind]
                oldclazz = self.classvec[flipindex]
                if self.classcounts[oldclazz] > self.constraint: 
                    self.flipClass(flipindex, clazz)
                    takecount += 1
                    if takecount >= min(howmany, len(takelist)): break
                takeind += 1
        
        #for clazz in range(self.labelcount):
        #    print self.computeFitnessForOneClass(self.VTY[:,clazz])

        self.updateA()




class RandomLabelSource(object):
    
    def __init__(self, size, labelcount):
        self.rand = Random()
        self.rand.seed(100)
        self.Y = - ones((size, labelcount), dtype = float64)
        self.classvec = - ones((size), dtype = int32)
        allinds = set(range(size))
        self.classcounts = zeros((labelcount), dtype = int32)
        for i in range(labelcount-1):
            inds = self.rand.sample(allinds, size / labelcount) #sampling without replacement
            allinds = allinds - set(inds)
            for ind in inds:
                self.Y[ind, i] = 1.
                self.classvec[ind] = i
                self.classcounts[i] += 1
        for ind in allinds:
            self.Y[ind, labelcount - 1] = 1.
            self.classvec[ind] = labelcount - 1
            self.classcounts[labelcount - 1] += 1
    
    def readLabels(self):
        return self.Y

