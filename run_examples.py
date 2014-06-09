import subprocess
import sys

exnames = [
'classacc_train',
'classacc_predict',
'classacc_performance',
'classacc_all',
'classAUC_train',
'classAUC_predict',
'classAUC_performance',
'classAUC_all',
'rankqids_train',
'rankqids_predict',
'rankqids_performance',
'rankqids_all',
'reg_train',
'reg_predict',
'reg_performance',
'reg_all',
'clustering',
'fselection',
'gaussian_kernel',
'polynomial_kernel',
'reduced_set',
'custom_folds',
'validationset',
'reduced_set_linear_kernel',
'reduced_set_linear_kernel_with_bias',
'cgrank_test',
'cgrank_test_with_preferences',
'cgrank_qids',
'cgrls_test',
]

exnames_doc = [
'classacc_all',
'classAUC_all',
'rankqids_all',
'reg_all',
'clustering',
'fselection',
'gaussian_kernel',
'polynomial_kernel',
'reduced_set',
'custom_folds',
'validationset',
'reduced_set_linear_kernel',
'reduced_set_linear_kernel_with_bias',
'cgrank_test',
'cgrank_test_with_preferences',
'cgrank_qids',
'cgrls_test',
]

#cfgs = ['examples/cfgs/cgrank_test.cfg',
#'examples/cfgs/cgrank_test_with_preferences.cfg',
#'examples/cfgs/cgrank_qids.cfg',
#'examples/cfgs/cgrls_test.cfg',
#'examples/cfgs/classacc_train.cfg',
#'examples/cfgs/classacc_predict.cfg',
#'examples/cfgs/classacc_performance.cfg',
#'examples/cfgs/classacc_all.cfg',
#'examples/cfgs/classacc_all_zero_bias.cfg',
#'examples/cfgs/classAUC_train.cfg',
#'examples/cfgs/classAUC_predict.cfg',
#'examples/cfgs/classAUC_performance.cfg',
#'examples/cfgs/classAUC_all.cfg',
#'examples/cfgs/clustering.cfg',
#'examples/cfgs/custom_folds.cfg',
#'examples/cfgs/develset.cfg',
#'examples/cfgs/fselection.cfg',
#'examples/cfgs/gaussian_kernel.cfg',
#'examples/cfgs/polynomial_kernel.cfg',
#'examples/cfgs/rankqids_train.cfg',
#'examples/cfgs/rankqids_predict.cfg',
#'examples/cfgs/rankqids_performance.cfg',
#'examples/cfgs/rankqids_all.cfg',
#'examples/cfgs/reduced_set.cfg',
#'examples/cfgs/reduced_set_linear_kernel.cfg',
#'examples/cfgs/reduced_set_linear_kernel_with_bias.cfg',
#'examples/cfgs/reg_train.cfg',
#'examples/cfgs/reg_predict.cfg',
#'examples/cfgs/reg_performance.cfg',
#'examples/cfgs/reg_all.cfg',
#]

if __name__=="__main__":
    
    for ename in exnames:
            #path = 'examples/cfgs/' + ename + '.cfg'
            path = 'examples/code/' + ename + '.py'
            print "*****"
            print path
            print "*****"
            sys.stdout.flush()
            sys.stderr.flush()
            #p = subprocess.Popen(['python', 'rls_core.py', path], stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stdout)
            p = subprocess.Popen(['python', path], stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stdout)
            sys.stdout.flush()
            sys.stderr.flush()
            p.wait()
