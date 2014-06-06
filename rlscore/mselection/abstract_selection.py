import numpy as np

from rlscore.utilities import array_tools


class AbstractSelection(object):
    """Abstract base class for creating new model selection strategies, such as those based on
    cross-validation"""
    
    
    def createMSelector(cls, **kwargs):
        selector = cls()
        selector.verbose = True
        selector.resource_pool = kwargs
        selector.setParameters(kwargs)
        selector.learner = kwargs['learner']
        selector.measure = kwargs['measure']
        selector.loadResources()
        return selector
    createMSelector = classmethod(createMSelector)
    
    
    def __init__(self):
        self.models = None
        self.performances = None
        self.verbose = True
        self.predictions = []
    
    
    def loadResources(self):
        """
        Loads the resources from the previously set resource pool.
        """
        if not self.resource_pool.has_key('train_labels'):
            raise Exception("ModelSelection cannot be initialized without labels in resource pool")
        Y = self.resource_pool['train_labels']
        self.Y = array_tools.as_labelmatrix(Y)
    
    
    def estimatePerformance(self, model):
        """Estimates the expected performance of a given model. This function should be overriden by the inheriting class
        
        @param model: trained learner object
        @type model: RLS
        @return: estimated performance for the model
        @rtype: float"""
        return 0.
    
    
    def setParameters(self, parameters):
        if parameters.has_key("reggrid"):
            #The reggrid may be a list, or string
            reggrid = parameters["reggrid"]
            if isinstance(reggrid, str):
                if "_" in reggrid:
                    reggrid = reggrid.split("_")
                    if len(reggrid) == 2 and is_int(reggrid[0]) and is_int(reggrid[1]):
                        lower = int(reggrid[0])
                        upper = int(reggrid[1])
                        if lower >= upper:
                            raise Exception("Maformed parameter grid: start point %d is not smaller than end point %d" %(lower, upper))
                        reggrid = range(lower, upper + 1)
                        reggrid = [2 ** x for x in reggrid]
                    else:
                        raise Exception("Malformed parameter grid: use e.g. '-5_5' to grid from 2^-5 to 2^5, or altenatively '1 10 100' type of grid")
                else:
                    reggrid = reggrid.split()
                    for value in reggrid:
                        if not is_number(value):
                            raise Exception("Malformed parameter grid: use e.g. '-5_5' to grid from 2^-5 to 2^5, or altenatively '1 10 100' type of grid")
                    reggrid = [float(x) for x in reggrid]      
            elif not isinstance(reggrid, list):
                raise Exception("Reggrid must be a list or a string")
        else:
            reggrid = range(-5, 6)
            reggrid = [2. ** x for x in reggrid]
        self.reggrid = reggrid 
    
    
    def reggridSearch(self):
        """Searches the regularization parameter grid to choose the value which, according to
        the estimatePerformance function, seems to provide best performance"""
        #Current assumption is that all of the algorithms included in the package will be based on regularized
        #risk minimization
        self.performances = []
        self.best_performance = None
        self.best_model = None
        self.best_regparam = None
        measure_name = str(self.measure).split()[1]
        if self.verbose:
            print "Regularization parameter grid initialized to", self.reggrid
        for regparam in self.reggrid:
            if self.verbose:
                print "Solving %s for regularization parameter value %f" % ("learner", regparam)
            self.learner.solve(regparam)
            performance = self.estimatePerformance(self.learner)
            self.performances.append(performance)
            if self.best_performance==None:
                self.best_performance = performance
                self.best_model =  self.learner.getModel()
                self.best_regparam = regparam
            else:
                #if compare_performances(self.measure, performance, self.best_performance) > 0:
                #if self.measure.comparePerformances(performance, self.best_performance) > 0:
                if (self.measure.iserror == (performance < self.best_performance)):
                    self.best_performance = performance
                    self.best_model = self.learner.getModel()
                    self.best_regparam = regparam
            if self.verbose:
                if performance != None:
                    print "%f %s (averaged), %f regularization parameter" % (performance, measure_name, regparam)
                else:
                    print "Performance undefined for %f regularization parameter" %regparam
        if self.verbose:
            if self.best_performance != None:
                print "Best performance %f %s with regularization parameter %f" % (self.best_performance, measure_name, self.best_regparam)
            else:
                print "Performance undefined for all tried values"
        self.resource_pool['mselection_performances'] = np.array([self.reggrid, self.performances]).T
        #some model selection strategies support this
        self.resource_pool['mselection_predictions'] = self.predictions
    
    
    def findBestModel(self):
        """Searches for the best model"""
        self.reggridSearch()
    
    
    def getBestModel(self):
        """Returns the best model
        @return: best model
        @rtype: RLS"""
        return self.best_model


def is_number(s):
    #checks whether a string represents a real number
    try:
        float(s)
    except ValueError:
        return False
    return True


def is_int(s):
    #checks whether a string represents an integer
    return s.isdigit() or (s[0]=="-" and s[1:].isdigit())

