from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy as np
import sys

USE_CYTHON = False
ext = '.pyx' if USE_CYTHON else '.c'

#sys.argv[1:] = ['build_ext', '--inplace']

ext_modules = [
    Extension("rlscore.utilities.swapped",["rlscore/utilities/swapped"+ext]),
    Extension("rlscore.learner.cython_pairwise_cv_for_rls",["rlscore/learner/cython_pairwise_cv_for_rls"+ext]),
    Extension("rlscore.learner.cython_pairwise_cv_for_global_rankrls",["rlscore/learner/cython_pairwise_cv_for_global_rankrls"+ext]),
    Extension("rlscore.learner.cython_two_step_rls_cv",["rlscore/learner/cython_two_step_rls_cv"+ext]),
    Extension("rlscore.learner.cython_mmc",["rlscore/learner/cython_mmc"+ext]),
    Extension("rlscore.learner.cython_greedy_rls",["rlscore/learner/cython_greedy_rls"+ext]),
    Extension("rlscore.utilities._sampled_kronecker_products",["rlscore/utilities/_sampled_kronecker_products"+ext])
]

if USE_CYTHON:
    from Cython.Build import cythonize
    ext_modules = cythonize(ext_modules)

setup(
    name = 'RLScore',
    url = "https://github.com/aatapa/RLScore",
    version = "0.93",
    license = "MIT",
    #setup_requires=["numpy"],
    #install_requires=["numpy", "scipy"],
    include_dirs = [np.get_include()],
    ext_modules = ext_modules,
    packages = find_packages(),
    )


