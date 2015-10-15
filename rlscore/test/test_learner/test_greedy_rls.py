from numpy import *
random.seed(100)
import numpy.linalg as la
import unittest

from rlscore.learner import GreedyRLS

    
    
def speedtest():
    tsize, fsize = 3000, 3000
    desiredfcount = 5
    Xtrain = mat(random.rand(fsize, tsize), dtype=float64)
    #Xtrain = mat(random.randint(0,10,size = (fsize, tsize)), dtype=int8)
    #save("foo",Xtrain)
    bias = 2.
    rp = 1.
    bias_slice = sqrt(bias)*mat(ones((1,Xtrain.shape[1]), dtype=float64))
    Xtrain_biased = vstack([Xtrain,bias_slice])
    #K = Xtrain.T * Xtrain
    ylen = 2
    #Y = mat(zeros((tsize, ylen), dtype=floattype))
    Y = mat(random.rand(tsize, ylen), dtype=float64)
    
    rpool = {}
    class TestCallback(object):
        def callback(self, learner):
            #print learner.performances[len(learner.performances)-1]
            #print 'GreedyRLS', learner.looperf.T
            print 'round'
        def finished(self, learner):
            pass
    tcb = TestCallback()
    rpool['callback'] = tcb
    rpool['X'] = Xtrain.T
    rpool['Y'] = Y
    
    rpool['subsetsize'] = str(desiredfcount)
    rpool['regparam'] = rp
    rpool['bias'] = bias
    grls = GreedyRLS(**rpool)
    
    print grls.selected
    print grls.A[grls.selected]
    print grls.b



class Test(unittest.TestCase):
    
    
    def testRLS(self):
        print
        print
        print
        print
        print "Testing the correctness of the GreedyRLS module."
        print
        print
        floattype = float64
        
        #m, n = 10, 30
        tsize, fsize = 10, 30
        desiredfcount = 5
        Xtrain = mat(random.rand(fsize, tsize), dtype=float64)
        #Xtrain = mat(random.randint(0,10,size = (fsize, tsize)), dtype=int8)
        #save("foo",Xtrain)
        #print Xtrain
        bias = 2.
        bias_slice = sqrt(bias)*mat(ones((1,Xtrain.shape[1]), dtype=float64))
        Xtrain_biased = vstack([Xtrain,bias_slice])
        #K = Xtrain.T * Xtrain
        ylen = 2
        #Y = mat(zeros((tsize, ylen), dtype=floattype))
        Y = mat(random.rand(tsize, ylen), dtype=float64)
        #Y = mat(random.randint(0,10,size = (tsize, 2)), dtype=int8)
        #save("bar",Y)
        #print Y
        #for i in range(tsize):
        #    if Y[i,0] < 0.5: Y[i,0] = -1.
        #    else: Y[i,0] = 1.
        
        selected = []
        
        rp = 1.
        currentfcount=0
        while currentfcount < desiredfcount:
            
            selected_plus_bias = selected + [fsize]
            bestlooperf = 9999999999.
            K = Xtrain_biased[selected_plus_bias].T*Xtrain_biased[selected_plus_bias] #+ mat(ones((tsize,tsize)))
            
            for ci in range(fsize):
                if ci in selected_plus_bias: continue
                cv = Xtrain_biased[ci]
                updK = Xtrain_biased[selected_plus_bias+[ci]].T*Xtrain_biased[selected_plus_bias+[ci]] #+ mat(ones((tsize,tsize)))
                #print 1. / diag(updG)
                looperf = 0.
                #'''
                for hi in range(tsize):
                    hoinds = range(0, hi) + range(hi + 1, tsize)
                    updcutK = updK[ix_(hoinds, hoinds)]
                    updcrossK = updK[ix_([hi], hoinds)]
                    loopred = updcrossK * la.inv(updcutK + rp * mat(eye(tsize-1))) * Y[hoinds]
                    looperf += mean(multiply((loopred - Y[hi]), (loopred - Y[hi])))
                '''
                loodiff = zeros((tsize, ylen))
                updG = la.inv(updK+rp * mat(eye(tsize)))
                for hi in range(tsize):
                    updcrossK = updK[hi]
                    loopred = updcrossK * updG * Y #THIS IS TRAINING SET ERROR, NOT LOO!!!
                    looperf += mean(multiply((loopred - Y[hi]), (loopred - Y[hi])))
                    loodiff[hi] = loopred - Y[hi]
                print loodiff.T'''
                if looperf < bestlooperf:
                    bestcind = ci
                    bestlooperf = looperf
                print 'Tester ', ci, looperf
            
            selected.append(bestcind)
            print 'Tester ', selected
            currentfcount += 1
        
        selected_plus_bias = selected + [fsize]
        K = Xtrain_biased[selected_plus_bias].T*Xtrain_biased[selected_plus_bias]
        G = la.inv(K+rp * mat(eye(tsize)))
        A = Xtrain_biased[selected_plus_bias]*G*Y
        print 'Tester ', A
        #A = mat(eye(fsize+1))[:,selected_plus_bias]*(Xtrain_biased[selected_plus_bias]*A)
        
        
        rpool = {}
        class TestCallback(object):
            def callback(self, learner):
                #print learner.performances[len(learner.performances)-1]
                print 'GreedyRLS', learner.looperf.T
                pass
            def finished(self, learner):
                pass
        tcb = TestCallback()
        rpool['callback'] = tcb
        rpool['X'] = Xtrain.T
        rpool['Y'] = Y
        #rpool['multi_task_X'] = [Xtrain.T,Xtrain.T]
        #rpool['multi_task_Y'] = [Y[:,0], Y[:,1]]
        
        rpool['subsetsize'] = desiredfcount
        rpool['regparam'] = rp
        rpool['bias'] = bias
        grls = GreedyRLS(**rpool)
        #grls = MTGreedyRLS(**rpool)
        print grls.selected
        print grls.A[grls.selected]
        print grls.b
        #for t in range(len(grls.alltasks)):
        #    print grls.alltasks[t].A[grls.selected]
        #    print grls.alltasks[t].b
        

if __name__=="__main__":
    #import cProfile
    #cProfile.run('speedtest()')
    suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    unittest.TextTestRunner(verbosity=2).run(suite)

