import sys
readers = {"folds":"read_folds", "train_features":"read_sparse", "test_features":"read_sparse", "train_qids":"read_qids", "test_qids":"read_qids"}

def generate(learner, lpath, lparams, lfparams, files, measure=None, selector=None, sparams=None):
    #learner: module for the learning algorithm
    #lparams: parameters for the learning algorithm
    #files: file:path pairs
    #
    code = []
    code.append("import numpy as np")
    code.append("from %s import %s" %(lpath, learner))
    for key in files:
        if key in readers:
            code.append("from rlscore.reader import %s" %readers[key])
    if measure != None:
        code.append("from rlscore.measure import %s" %measure)
    if selector != None:
        code.append("from %s import %s" %(lpath, selector))
        code.append("from rlscore.utilities.grid_search import grid_search")
    for key in files:
        if key in readers:
            code.append('%s = %s("%s")' %(key, readers[key], files[key]))
        else:
            code.append('%s = np.loadtxt("%s")' %(key, files[key])) 
    code.append("kwargs = {}")
    for key in lfparams:
        if key == "basis_vectors":
            code.append('kwargs["%s"] = '%key +"train_features["+str(lfparams[key])+"]")
        else:
            code.append('kwargs["%s"] = '%key +str(lfparams[key]))
    for key in lparams:
        if isinstance(lparams[key], str):
            code.append('kwargs["%s"] = "%s"' %(key, lparams[key]))
        else:
            code.append('kwargs["%s"] = '%key +str(lparams[key]))
    code.append("learner = %s.createLearner(**kwargs)" %learner)
    code.append("learner.train()")
    #If model selection
    if selector:
        code.append("kwargs = {}")
        code.append('kwargs["learner"] = learner')
        for key in sparams:
            code.append('kwargs["%s"] = '%key +str(sparams[key]))        
        code.append("crossvalidator = %s(**kwargs)" %selector)
        code.append("grid = [2**i for i in range(-10,11)]")
        code.append("learner, perfs = grid_search(crossvalidator, grid)")
        code.append("for i in range(len(grid)):")
        code.append('    print "parameter %f cv_performance %f" %(grid[i], perfs[i])')
    code.append("model = learner.getModel()")
    if "test_features" in files.keys():
        code.append("P = model.predict(test_features)")
        if measure != None and "test_labels" in files.keys():
            if "test_qids" in files.keys():
                code.append("from rlscore.measure.measure_utilities import UndefinedPerformance")
                code.append("perfs = []")
                code.append("for query in test_qids:")
                code.append("    try:")
                code.append("        perf = %s(test_labels[query], P[query])" %measure)
                code.append("        perfs.append(perf)")
                code.append("    except UndefinedPerformance:")
                code.append("        pass")
                code.append("test_perf = np.mean(perfs)") 
            else:
                code.append("test_perf = %s(test_labels, P)" %measure)
            code.append('print "test set performance: %f" %test_perf')
    return code

def generate_all():
    import rls_experiments
    import rankrls_experiments
    import greedy_experiments
    import clustering_experiments
    for module in [rls_experiments, rankrls_experiments, greedy_experiments, clustering_experiments]:
        experiments = module.experiments
        for key in experiments:
            params = experiments[key]
            code = generate(**params)
            f = open("../code/%s.py" %key, 'w')
            for line in code:
                f.write(line+"\n")
            f.close()
        
if __name__=="__main__":
    generate_all()
    #for line in code:
    #    print line
    #generate_all()
    
    
    

    
