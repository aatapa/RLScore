from distutils.core import setup
from distutils.extension import Extension
from distutils.command import build_ext

ext_modules = [Extension(
    name="rlscore.utilities.swapped",
    sources=["ext_src/swapped.c", "ext_src/c_swapped.c"],
    language="c"),
    Extension("rlscore.learner.cython_mmc",["rlscore/learner/cython_mmc.c"]),
    Extension("rlscore.learner.cython_greedy_rls",["rlscore/learner/cython_greedy_rls.c"]),
    Extension("rlscore.utilities.sparse_kronecker_multiplication_tools",["rlscore/utilities/sparse_kronecker_multiplication_tools.c"])]

setup(
    name = 'RLScore',
    ext_modules = ext_modules,
    )

