from distutils.core import setup
from distutils.extension import Extension
import numpy as np
import sys

sys.argv[1:] = ['build_ext', '--inplace']

ext_modules = [
    Extension("rlscore.utilities.swapped",["rlscore/utilities/swapped.c"], include_dirs=[np.get_include()] ),
    Extension("rlscore.learner.cython_pairwise_cv_for_rls",["rlscore/learner/cython_pairwise_cv_for_rls.c"], include_dirs=[np.get_include()]),
    Extension("rlscore.learner.cython_pairwise_cv_for_global_rankrls",["rlscore/learner/cython_pairwise_cv_for_global_rankrls.c"], include_dirs=[np.get_include()]),
    Extension("rlscore.learner.cython_mmc",["rlscore/learner/cython_mmc.c"], include_dirs=[np.get_include()]),
    Extension("rlscore.learner.cython_greedy_rls",["rlscore/learner/cython_greedy_rls.c"], include_dirs=[np.get_include()]),
    Extension("rlscore.utilities._sampled_kronecker_products",["rlscore/utilities/_sampled_kronecker_products.c"], include_dirs=[np.get_include()])
]

setup(
    name = 'RLScore',
    ext_modules = ext_modules,
    )


