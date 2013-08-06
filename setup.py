from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension(
    name="rlscore.utilities.swapped",
    sources=["ext_src/swapped.pyx", "ext_src/c_swapped.c"],
    language="c"),
    Extension("rlscore.learner.cython_mmc",["rlscore/learner/cython_mmc.pyx"]),
    Extension("rlscore.learner.cython_greedy_rls",["rlscore/learner/cython_greedy_rls.pyx"]),
    Extension("rlscore.utilities.sparse_kronecker_multiplication_tools",["rlscore/utilities/sparse_kronecker_multiplication_tools.pyx"])]

setup(
    name = 'cmodules',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    )

