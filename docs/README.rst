=======
RLScore
=======


RLScore - regularized least-squares machine learning algorithms package.


:Authors:         `Tapio Pahikkala <http://staff.cs.utu.fi/~aatapa/>`_,
                  `Antti Airola <https://scholar.google.fi/citations?user=5CPOSr0AAAAJ>`_
:Email:           firstname.lastname@utu.fi
:Homepage:        `http://staff.cs.utu.fi/~aatapa/software/RLScore <http://staff.cs.utu.fi/~aatapa/software/RLScore>`_
:Version:         0.8.1
:License:         `The MIT License <LICENCE.TXT>`_
:Date:            August 22. 2018

.. contents::

Overview
========

RLScore is a machine learning software package for regularized kernel methods,
focusing especially on Regularized Least-Squares (RLS) based methods. The main
advantage of the RLS family of methods is that they admit a closed form solution, expressed as a system of linear equations.
This allows deriving highly efficient algorithms for RLS methods, based on matrix
algebraic optimization. Classical results include computational short-cuts for
multi-target learning, fast regularization path and leave-one-out
cross-validation. RLScore takes these results further by implementing a wide
variety of additional computational shortcuts for different types of cross-validation
strategies, single- and multi-target feature selection, multi-task and zero-shot
learning with Kronecker kernels, ranking, stochastic hill climbing based
clustering etc. The majority of the implemented methods are such that are not
available in any other software package.

For documentation, see project `home page <http://staff.cs.utu.fi/~aatapa/software/RLScore>`_.


Support for different tasks
===========================


-  Regression and classification
   
   - Regularized least-squares (RLS)
       - multi-target learning
       - regularization path
       - leave-one-out cross-validation
       - leave-pair-out cross-validation
       - fast cross-validation with arbitrary hold-out sets
   
-  Feature selection for regression and classification

   - Greedy regularized least-squares (Greedy RLS)
       - greedy forward selection; selects features based on leave-one-out error
       - joint feature selection for multi-target problems
       
-  Ranking

   - Regularized least-squares ranking (GlobalRankRLS)
       - minimizes magnitude preserving ranking error
       - multi-target learning
       - regularization path
       - leave-pair-out cross-validation
       - cross-validation with arbitrary hold-out sets

   - Regularized least-squares ranking for query-structured data (QueryRankRLS)
       - minimizes magnitude preserving ranking error, computed for each query separately
       - multi-target learning
       - regularization path
       - leave-query-out cross-validation
       
-  Pair-input data and zero-shot learning

   - Learning with Kronecker product kernels
       - Closed form solution for training models from complete data with labels for all pair-inputs available (KronRLS, TwoStepRLS)
       - Leave-one-out and k-fold cross-validation algorithms for pair-input data (TwoStepRLS)
       - Iterative training algorithm for pair-input data, where only a subset of pairwise labels are known (CGKronRLS)

-  Clustering

   - Unsupervised RLS methods, based on the maximum margin clustering principle


Software dependencies
=====================

RLScore is written in Python and thus requires a working
installation of Python 3.5 or newer. The package is also dependent on
the `NumPy <http://numpy.scipy.org/>`_ package for matrix
operations, and `SciPy <http://www.scipy.org/>`_ package for sparse
matrix implementations, and a c-compiler for building Cython extensions.

Citing RLScore
==============

RLScore is described in the following article:

`Rlscore: Regularized least-squares learners <http://jmlr.org/papers/v17/16-470.html>`_, Tapio Pahikkala and Antti Airola. Journal of Machine Learning Research, 17(221):1-5, 2016. BibTeX entry can be found `here <http://jmlr.org/papers/v17/16-470.bib>`_.


History
=======

Version 0.8.1 (2018.08.22):

- New tutorials for stacked (two-step) kernel ridge regression e.g. two-step RLS
- Many technical improvements for learning with pairwise data
- Requires Python 3.5 or newer due to the use of matrix product infix notation etc.

Version 0.8 (2017.08.17):

- Compatible with Python 3
- Last version still working properly with Python 2.7

Version 0.7 (2016.09.19):

- Tutorials available
- API documentation finished
- TwoStep-learning cross-validation methods available
- Unit testing extended
- Simplified internal structure of the package

Version 0.6 (2016.02.18):

- Major overhaul of learner interface, leaners now trained directly when initialized
- TwoStep-learning method, better Kronecker learners
- Cythonization of leave-pair-out cross-validation
- Automated regularization parameter selection via cross-validation for RLS and RankRLS added
- Old documentation removed as out-of-date, new documentation and tutorials in preparation

Version 0.5.1 (2014.07.31):

- This is a work in progress version maintained in a github repository.
- The command line functionality is dropped and the main focus is shifted towards the library interface.
- The interface has been considerably simplified to ease the use of the library.
- Learning with tensor (Kronecker) product kernels considerably extended.
- Many learners now implemented with cython to improve speed.
- Support for a new type of interactive classification usable for image segmentation and various other tasks.
- Numerous internal changes in the software.

Version 0.5 (2012.06.19):

- CGRLS and CGRankRLS learners for conjugate gradient -based training of RLS/RankRLS on large and high-dimensional, but sparse data.
- CGRankRLS supports learning from pairwise preferences between data points in addition to learning from utility values.
- Library interface for Python. Code examples for almost all included learning algorithms.
- Support for learning with Kronecker kernels.
- Numerous internal changes in the software.

Version 0.4 (2010.04.14):

- A linear time greedy forward feature selection with leave-one-out criterion for RLS (greedy RLS) included.
- Example data and codes for basic use cases included in the distribution.
- Fixed a bug causing problems when reading/writing binary files in Windows.
- Modifications to the configuration file format.
- All command line interfaces other than rls_core.py removed.

Version 0.3 (2009.12.03):

- Major restructuring of the code to make the software more modular.
- Configuration files introduced for more flexible use of software.
- Evolutionary maximum-margin clustering included.
- Model file format changed.

Version 0.2.1 (2009.06.24):

- Fixed a bug causing one of the features to get ignored.

Version 0.2 (2009.03.13):

- Major overhaul of the file formats.
- RLScore now supports learning multiple tasks simultaneously.
- Reduced set approximation included for large scale learning.

Version 0.1.1 (2009.01.11):

- Fixed a bug causing a memory leak after training with sparse data and linear kernel.

Version 0.1 (2008.10.18):

- First public release.

Credits
=======

:Other Contributors: 

`Michiel Stock  <https://michielstock.github.io/>`_
                provided code for fast cross-validation with stacked (two-step) kernel ridge regression (version 0.8.1)

`Evgeni Tsivtsivadze <http://learning-machines.com/>`_
                      participated in designing the version 0.1
                        
                       







