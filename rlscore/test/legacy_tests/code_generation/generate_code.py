import sys
readers = {"folds":"read_folds", "train_features":"read_sparse", "test_features":"read_sparse", "train_qids":"read_qids", "test_qids":"read_qids"}

def generate(learner, lpath, lparams, lfparams, files, measure=None, selection=False):
    #learner: module for the learning algorithm
    #lparams: parameters for the learning algorithm
    #files: file:path pairs
    #
    code = []
    code.append("import numpy as np")
    code.append("from %s import %s" %(lpath, learner))
    for key in files:
        if key in readers:
            code.append("from rlscore.utilities.reader import %s" %readers[key])
    if measure is not None:
        code.append("from rlscore.measure import %s" %measure)
    for key in files:
        if key in readers:
            code.append('%s = %s("%s")' %(key, readers[key], files[key]))
        else:
            code.append('%s = np.loadtxt("%s")' %(key, files[key])) 
    code.append("kwargs = {}")
    if measure is not None and selection:
        code.append("kwargs['measure']=%s" %measure)
    if selection:
        code.append("kwargs['regparams'] = [2**i for i in range(-10,11)]")
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
    code.append("learner = %s(**kwargs)" %learner)
    #If model selection
    if selection:
        code.append("grid = kwargs['regparams']")
        code.append("perfs = learner.cv_performances")
        code.append("for i in range(len(grid)):")
        code.append('    print "parameter %f cv_performance %f" %(grid[i], perfs[i])')
    if "test_features" in files.keys():
        code.append("P = learner.predict(test_features)")
        if measure is not None and "test_labels" in files.keys():
            if "test_qids" in files.keys():
                code.append("from rlscore.measure.measure_utilities import UndefinedPerformance")
                code.append("from rlscore.measure.measure_utilities import qids_to_splits")
                code.append("test_qids = qids_to_splits(test_qids)") 
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
    
    
    

    
