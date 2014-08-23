=======
RLScore
=======


RLScore - regularized least-squares based machine learning algorithms
for regression, classification, ranking, clustering, and feature selection.


:Authors:         `Tapio Pahikkala <http://staff.cs.utu.fi/~aatapa/>`_,
                  `Antti Airola <http://tucs.fi/education/phd/alumni/student.php?student=120>`_
:Email:           firstname.lastname@utu.fi
:Homepage:        `http://staff.cs.utu.fi/~aatapa/software/RLScore/ <http://staff.cs.utu.fi/~aatapa/software/RLScore/>`_
:Version:         0.5.1
:License:         `The MIT License <LICENCE.TXT>`_
:Date:            2012.06.19


.. contents::


Overview
========

RLScore is a Regularized Least-Squares (RLS) based algorithm
package. It contains implementations of the RLS and RankRLS
learners allowing the optimization of performance measures for the
tasks of regression, ranking and classification. In addition,
the package contains  linear time greedy forward feature selection
with leave-one-out criterion for RLS (greedy RLS). Finally, 
the package contains an implementation of a maximum margin
clustering method based on RLS and stochastic hill climbing.
Implementations of efficient cross-validation algorithms are
integrated to the package, combined together with functionality
for fast parallel learning of multiple outputs.

Reduced set approximation for large-scale learning with kernels is
included. In this setting approximation is introduced also to the
cross-validation methods. For learning linear models from large but
sparse data sets, RLS and RankRLS can be trained using conjugate
gradient optimization techniques.

Support for different tasks
===========================


-  Regression
   
   -  Directly optimizes for the squared loss using the RLS algorithm
   -  fast n-fold cross-validation algorithm, where folds of arbitrary
      sizes can be left out of the training set
   -  fast leave-one-out (LOO) cross-validation algorithm
   -  efficient searching for the regularization parameter

-  Classification
   
   -  optimizes for accuracy using the RLS algorithm
   -  optimizes for AUC (Area Under the ROC Curve) using the RankRLS
      algorithm
   -  fast n-fold cross-validation algorithm, where folds of arbitrary
      sizes can be left out of the training set
   -  fast leave-one-out (LOO) cross-validation algorithm for accuracy
      optimization
   -  fast leave-pair-out (LPO) cross-validation algorithm for AUC
      optimization
   -  efficient searching for the regularization parameter

-  Ranking

   -  directly optimizes for the squared magnitude preserving ranking
      loss using RankRLS algorithm
   -  optimizes for disagreement error using the RankRLS algorithm
   -  fast label ranking cross-validation algorithm (leave-query-out)
   -  fast leave-pair-out (LPO) cross-validation algorithm for object
      ranking

-  Clustering

   -  given an unlabeled set of data, performs a stochastic hill
      climbing based search for cluster labels, while using RLS as a
      criterion
   -  supports user-given preliminary cluster labels in addition to
      random ones

-  Feature selection

   -  the feature selection algorithm in RLScore, greedy RLS, can be
      used to learn sparse linear RLS predictors efficiently, that is,
      the time complexity of Greedy RLS is linear in the number of
      training data, the number of features in the original data
      set, and the desired size of the set of selected features.
   -  greedy RLS starts from an empty feature set, and on each
      iteration adds the feature whose addition provides the best
      leave-one-out cross-validation performance.


Download
========

Download `RLScore.zip <RLScore.zip>`_ containing the python source code of RLScore.


Software dependencies
=====================

RLScore is written in Python and thus requires a working
installation of Python 2.8.x. The package is also dependent on
the `NumPy 1.3.x <http://numpy.scipy.org/>`_ package for matrix
operations, and `SciPy 0.13.x <http://www.scipy.org/>`_ package for sparse
matrix implementations.

Usage
=====
RLScore is designed to be used as a software library, called directly from Python code.

The easiest way to use RLScore is by modifying one of the example python codes delivered
with the distribution, to match your task. The software supports a wide variety of
different learning tasks, ranging from supervised learning to clustering and feature
selection. 


Examples
========

Examples of typical use-cases for each type of task are provided below.

The example code files, and the example data sets used by them can be found
in the 'examples' folder of the RLScore distribution. For example, to run the
example file 'rls_classification.py' included in examples/code from the command line,
go to the folder containing the RLScore distribution, and execute the command
'python examples/code/rls_classification.py'

While the examples use Unix-style paths with '/' separator,
they should also work in Windows with no modifications needed.


Binary classification, maximize accuracy
----------------------------------------

In binary classification, the data is separated into two classes. Classification
accuracy measures the fraction of correct classifications made by the learned
classifier. This is perhaps the most widely used performance measure for binary
classification. However, for very unbalanced data-sets it may be preferable to
optimize the area under ROC curve (AUC) measure, considered in a later example,
instead.

When training a classifier according to the accuracy criterion, using the
RLS module which minimizes a least squares loss on the training set class
labels is recommended. The approach is equivalent to the so-called
least-squares support vector machine. 

Requirements:
- class labels should be either 1 (positive) or -1 (negative)


Python code (rls_classification)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: examples/code/rls_classification.py
   :literal:

Ranking with RankRLS, minimize pairwise mis-orderings
-----------------------------------------------------

In ranking the aim is to learn a function, whose predictions result in an
accurate ranking when ordering new examples according to the predicted
values. That is, more relevant examples should receive higher predicted
scores than less relevant.

When training a ranker, using the RankRLS learner which minimizes a pairwise
least-squares loss on the training set class labels is recommended.

Using qids means that instead of a total order over all examples, each
query has it's own ordering, and examples from different queries should
not be compared. For example in information retrieval, each query
might consist of the ordering of a set of documents according to
a query posed by a user.

There are two variants of the RankRLS module, the LabelRankRLS that should
be used when using queries, and the AllPairsRankRLS that should be used
otherwise.

Additionally, if the data is both high dimensional and sparse, one should use the module
CGRankRLS, which is optimized for such a data
(see `Learning linear models from large sparse data sets`_).

In addition to learning from utility scores of data points, CGRankRLS also
supports learning from pairwise preferences, see
`Python code (rankrls_cg_preferences)`_


Python code for query ranking (rankrls_lqo.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: examples/code/rankrls_lqo.py.py
   :literal:
   
Python code for global ranking (rankrls_lpo.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: examples/code/rankrls_lpo.py.py
   :literal:



Regression
----------

In regression, the task is to predict real-valued labels. The regularized
least-squares (RLS) module is suitable for solving this task.


Python code (rls_regression)
~~~~~~~~~~~~~~~~~~~~~

.. include:: examples/code/rls_regression.py
   :literal:



Clustering with evolutionary maximum margin clustering
------------------------------------------------------

In clustering, the task is to divide unlabeled data into several clusters.
One aims to find such cluster structure that within a cluster the data points are
similar to each other, but dissimilar with respect to the examples in the other
clusters. 

The clustering algorithm implemented in RLScore aims to divide the
data so that the resulting division yields minimal
regularized least-squares error. The approach is analogous to the maximum
margin clustering approach. The resulting combinatorial optimization
problem is NP-hard, stochastic hill-climbing together with computational
shortcuts is used to search for a locally optimal solution. Re-starts
may be necessary for discovering good clustering.


Python code (mmc_defparams)
~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: examples/code/mmc_defparams.py
   :literal:



Feature selection with greedy RLS
---------------------------------

GreedyRLS, the feature selection module of RLScore, allows selecting a
fixed size subset of features. The selection criterion is the performance
of a RLS learner when trained on the selected features, which is measured
using leave-one-out cross-validation. Both regression and classification
tasks are supported.

In addition to feature selection, the module can be used to train sparse
RLS predictors that use only a specified amount of features for making
predictions. Only linear learning is supported. The method scales
linearly with respect to the number of examples, features and selected
features.

The indices of the selected features are written to the file provided
as the 'selected_features' parameter.
The LOO performances made by GreedyRLS in each step of the greedy forward
selection process are written to the file provided
as the 'GreedyRLS_LOO_performances' parameter.


Python code (greedyrls)
~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: examples/code/greedyrls.py
   :literal:



Using kernels
-------------

Most of the learning algorithms included in the RLScore package support the
use of also other kernels than the linear one. Efficient implementations
for calculating the Gaussian and the polynomial kernel are included.

The training algorithms explicitly construct and decompose the full kernel
matrix, resulting in squared memory and cubic training complexity. Performing
cross-validation or multiple output learning does not increase this complexity
due to computational shortcuts. In practice kernels can be used with several
thousands of training data points. For large scale learning with kernels, see
reduced set approximation

Currently grid searching for kernel parameters is not supported, the
way to accomplish this is to write a wrapper script around rls_core.

In the following example we traing a RLS classifier using Gaussian kernel,
the other learners can be used with kernels in an analogous way. The only
change needed to the earlier examples is to define 'kernel=GaussianKernel'
and supply the kernel parameters under [Parameters].


Python code (rls_gaussian)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: examples/code/rls_gaussian.py
   :literal:


Python code (rls_polynomial)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: examples/code/rls_polynomial.py
   :literal:

Learning linear models from large sparse data sets
--------------------------------------------------

In settings where both the number of training data and the number
of features are large, but the data is sparse (most entries in data
matrix zeroes), regression, classification and ranking can be done
much more efficiently using the conjugate gradient training algorithms.
In this case, kernels are not supported, only linear models. The
methods allow substantial savings in memory usage and improved scaling,
since they need only the non-zero entries in the data
matrix for training, and avoid computing samples x samples or
features x features sized matrices.

In this setting, the CRGRLS module can be used analogously to the RLS
module, and the CGRankRLS module can be used analogously to
AllPairsRankRLS / LabelRankRLS. The CG-implementations do not support
cross-validation.

In addition to learning from utility scores of data points, CGRankRLS also
supports learning from pairwise preferences.

Python code (cg_rls)
~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: examples/code/cg_rls.py
   :literal:


Python code (rankrls_cg)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: examples/code/rankrls_cg.py
   :literal:


Python code (rankrls_cg_qids)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: examples/code/rankrls_cg_qids.py
   :literal:

Python code (rankrls_cg_preferences)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: examples/code/rankrls_cg_preferences.py
   :literal:



Reduced set approximation
-------------------------

Once training data set size exceeds several thousand examples, training the
learning methods with (non-linear) kernels becomes infeasible.
For this case RLScore implements the reduced set approximation algorithm, where
only a pre-specified subset of training examples are used to represent the dual
solution learned. 

To use the reduced set approximation, one should supply the indices of those
training examples which are used to represent the learned solution
(so-called 'basis 'vectors') in a file. The file should contain one line,
where the indices are separated with whitespaces.

The best way for selecting the basis vectors is an open research question,
uniform random subsampling of training set indices provides usually decent
results.

While cross-validation can be performed with the reduced set approximation,
the results are only approximative. For small regularization parameter
values pessimistic bias has been observed in the cross-validation estimates.


Python code (rls_reduced)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: examples/code/rls_reduced.py
   :literal:


History
=======

Version 0.5.1 (2014.07.31)
------------------------
- This is a work in progress version maintained in a github repository.
- The command line functionality is dropped and the main focus is shifted towards the library interface.
- The interface has been considerably simplified to ease the use of the library.
- Learning with tensor (Kronecker) product kernels considerably extended.
- Many learners now implemented with cython to improve speed.
- Support for a new type of interactive classification usable for image segmentation and various other tasks.
- Numerous internal changes in the software.

Version 0.5 (2012.06.19)
------------------------
- CGRLS and CGRankRLS learners for conjugate gradient -based training of RLS/RankRLS on large and high-dimensional, but sparse data.
- CGRankRLS supports learning from pairwise preferences between data points in addition to learning from utility values.
- Library interface for Python. Code examples for almost all included learning algorithms.
- Support for learning with Kronecker kernels.
- Numerous internal changes in the software.

Version 0.4 (2010.04.14)
------------------------

- A linear time greedy forward feature selection with leave-one-out criterion for RLS (greedy RLS) included.
- Example data and codes for basic use cases included in the distribution.
- Fixed a bug causing problems when reading/writing binary files in Windows.
- Modifications to the configuration file format.
- All command line interfaces other than rls_core.py removed.


Version 0.3 (2009.12.03)
------------------------

- Major restructuring of the code to make the software more modular.
- Configuration files introduced for more flexible use of software.
- Evolutionary maximum-margin clustering included.
- Model file format changed.

Version 0.2.1 (2009.06.24)
--------------------------

- Fixed a bug causing one of the features to get ignored.

Version 0.2 (2009.03.13)
------------------------

- Major overhaul of the file formats.
- RLScore now supports learning multiple tasks simultaneously.
- Reduced set approximation included for large scale learning.

Version 0.1.1 (2009.01.11)
--------------------------

- Fixed a bug causing a memory leak after training with sparse data and linear kernel.

Version 0.1 (2008.10.18)
------------------------

- First public release.

Credits
=======

:Former Contributors: `Evgeni Tsivtsivadze <http://learning-machines.com/>`_ -
                      participated in designing the version 0.1 and co-authored some
                      of the articles in which the implemented methods were proposed.







