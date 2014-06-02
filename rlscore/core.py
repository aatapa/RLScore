import sys
import numpy as np

from rlscore import data_sources
from rlscore import reader
from rlscore import writer
from rlscore import measure
from rlscore import mselection
from rlscore.measure import measure_utilities
from rlscore.utilities import creators
from rlscore.measure.measure_utilities import UndefinedPerformance

LEARNER_NAME = 'learner'
KERNEL_NAME = 'kernel'
MEASURE_NAME = 'measure'
MSELECTION_NAME = 'mselection'
CALLBACK_NAME = 'callback'

def loadCore(modules, parameters, input_file, output_file, input_reader = {}, output_writer = {}):
    """Runs RLScore.
    
    This interface is intended to be used by the rls_core program, that based on a configuration
    file runs the whole learning procedure:
    
    read input -> model selection -> training -> write outputs
    
    For using rlscore as part of your own program, it is recommended to rather to use the
    trainModel or createLearner - interfaces, or the lower level interfaces available in
    the individual modules.
    
    Parameters
    ----------
    modules: dictionary, {type : module} string pairs
        learner, kernel, mselection and measure
    parameters: dictionary, {parameter : value} string pairs
        parameters for the learner, kernel and mselection
    input_file: dictionary, {variable : path} string pairs
        variable-file pairs for input data
    output_file: dictionary, {variable : path} string pairs
        variable-file pairs for output data
    input_reader: dictionary, {variable : reader} string pairs, optional
        variable-reader pairs describing file readers
    input_writer: dictionary, {variable : writer} string pairs, optional
        variable-writer pairs describing file writers
    """
    
    rpool = {}
    if 'importpath' in parameters:
        paths = parameters['importpath'].split(";")
        for path in paths:
            sys.path.append(path)
        del parameters['importpath']
    for varname, fname in input_file.iteritems():
        vartype = data_sources.VARIABLE_TYPES[varname]
        rfunc = reader.DEFAULT_READERS[vartype]
        if varname in data_sources.COMPOSITES:
            varnames = data_sources.COMPOSITES[varname]
            reader.composite_to_rpool(rpool, fname, rfunc, varnames)
        else:
            rpool[varname] = rfunc(fname)
    
    rpool.update(parameters)
    rpool.update(modules)
    
    #dynamic imports of modules
    if MEASURE_NAME in modules:
        measurefun = eval("measure."+modules[MEASURE_NAME])
    else:
        measurefun = None
    rpool['measure'] = measurefun
    
    if CALLBACK_NAME in modules:
        exec "import utilities." + modules[CALLBACK_NAME]
        callback = eval("utilities." + modules[CALLBACK_NAME]).CallbackFunction()
        rpool[CALLBACK_NAME] = callback
    
    if LEARNER_NAME in modules and modules[LEARNER_NAME] != None:
        kwargs = trainModel(**rpool)
        rpool.update(kwargs)
    
    #Make predictions, if model and test examples available
    if rpool.has_key('model') and rpool.has_key('prediction_features'):
        print "Making predictions on test data"
        model = rpool['model']
        predictions = model.predictFromPool(rpool)
        rpool['predicted_labels'] = predictions
    
    #Measure performance, if predictions, true labels and performance measure available
    if measurefun != None and rpool.has_key('predicted_labels') and rpool.has_key('test_labels'):
        correct = rpool['test_labels']
        predicted = rpool['predicted_labels']
        if rpool.has_key('test_qids'):
            print "calculating performance as averages over queries"
            q_partition = rpool['test_qids']
            perfs = []
            for query in q_partition:
                try:
                    perf = measurefun(correct[query], predicted[query])
                    perfs.append(perf)
                except UndefinedPerformance, e:
                    pass
            performance = np.mean(perfs)
        else:
            performance = measurefun(correct, predicted)
        measure_name = str(measurefun).split()[1]
        print 'Performance: %f %s' % (performance, measurefun.__name__)
        rpool['test_performance'] = performance
    
    for varname, fname in output_file.iteritems():
        vartype = data_sources.VARIABLE_TYPES[varname]
        wfunc = writer.DEFAULT_WRITERS[vartype]
        wfunc(fname, rpool[varname])


def trainModel(**kwargs):
    learner = createLearner(**kwargs)
    kwargs[LEARNER_NAME] = learner
    if MSELECTION_NAME in kwargs:
        mselector = eval("mselection."+kwargs[MSELECTION_NAME]+".createMSelector(**kwargs)")
        mselector.findBestModel()
        #we have the most promising model
        model = mselector.getBestModel()
        kwargs.update(mselector.resource_pool)
    else:
        learner.train()
        if hasattr(learner, 'resource_pool'):
            kwargs.update(learner.resource_pool)
        if hasattr(learner, 'results'):
            kwargs.update(learner.results)
        model = learner.getModel()
    kwargs.update(learner.results)
    kwargs['model'] = model
    return kwargs


def createLearner(**kwargs):
    #if kwargs.has_key(KERNEL_NAME):
    #    kernel = creators.createKernelByModuleName(**kwargs)
    #    kwargs['kernel_obj'] = kernel
    learner = createLearnerByModuleName(**kwargs)
    return learner


def createLearnerByModuleName(**kwargs):
    
    lname = kwargs[LEARNER_NAME]
    exec "from learner import " + lname
    learnerclazz = eval(lname)
    learner = learnerclazz.createLearner(**kwargs)
    return learner
    

