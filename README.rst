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



[Parameters]
~~~~~~~~~~~~

Parameters section contains the parameters supplied to RLScore. The meaning of kernel and
learner parameters differs for different learning and kernel modules.

regparam
........

Supply a float valued regularization parameter if you wish to train a learner with a
pre-selected parameter value. This value is used, if no model selection module is defined.
Must be positive. The default value is 1.


reggrid
.......

Regularization parameter grid searched during model selection. The value of the attribute is given as lower_upper, where lower and upper must be integers, with upper > lower. The grid becomes 2**lower ... 2**upper, that is, all integer powers of 2 between 2**lower and 2**upper are tested as values of the regularization parameter and the one with the best performance is selected. The default grid is -5_5. Alternatively, all the parameter values in the grid can be given directly, e.g. '0.001, 0.1, 1, 10, 50'.

    
bias
....

Float valued bias term, that corresponds to a new constant-valued feature added to each data point.
Allows learning models of the type f(x)+b, where a constant value (learned from data) is added to
each prediction. The value must be positive, the default value is 0. Can be useful for RLS learners,
when using linear kernel and low-dimensional data.

number_of_clusters
..................

Parameter supplied to the MMC learner. Its value is an integer specifying the desired number of clusters.

subsetsize
..........

Parameter supplied to the GreedyRLS learner. Its value is an integer defining the number of selected features.

gamma
.....

Float valued positive kernel parameter for the Gaussian or the polynomial kernel. For the Gaussian kernel k(x,z) = e^(-gamma*<x-z,x-z>), for polynomial kernel  k(x,z) = (gamma * <x,z> + coef0)^degree. (default = 1.).

coef0
.....

Float valued kernel parameter for the polynomial kernel. k(x,z) = (gamma * <x,z> + coef0)^degree. (default = 0)

degree
......

Integer valued positive kernel parameter for the polynomial kernel. k(x,z) = (gamma * <x,z> + coef0)^degree. (default = 2)

[Input]
~~~~~~~

The attributes in this section are names of `RLScore variables`_ used inside the RLScore software. The values of the attributes are filenames from which data is loaded to the variables. For example, the feature representations of the training data are loaded into a variable of name `train_features`_. Some of the loaded `[Modules]`_ require certain valiables to be loaded. The loaded variables also have an effect on what rls_core does.
 
All variables have their corresponding default file formats. Detailed descriptions of the variables and their default file formats are given in `RLScore variables`_.

    
[Output]
~~~~~~~~

Analogously to the `[Input]`_ section, the attributes in this section are names of variables used inside RLScore. The values of the attributes are names of files into which the contents of the variable are written to. The files are written in the default format of the variable in question.


RLScore variables
-----------------

RLScore variables are used to refer to the different types of data inside the RLScore software. The contents of the variables can be loaded from a file via the `[Input]`_ section or they are generated by the software itself. For example, if the contents of the `model`_ and `prediction_features`_ variables are provided, the software uses the model to perform predictions for the data points represented by the `prediction_features`_ variable and the predictions are put to the variable `predicted_labels`_. The contents of `predicted_labels`_ can then be saved into file or used for performance evaluation if the contents of the `test_labels`_ variable are also provided.


train_features
~~~~~~~~~~~~~~

Variable containing features for training data. The default file format is the one described in `Featurefile`_.


train_labels
~~~~~~~~~~~~

Variable containing labels for training data. Necessary when training supervised learners. The default file format is the one described in `Labelfile`_.

train_qids
~~~~~~~~~~

Qids for the training data. The default file format is the one described in `Qid file`_.

basis_vectors
~~~~~~~~~~~~~

Use reduced set approximation to speed up training and prediction. Restricts the learned hypothesis to be represented only by the training data points whose indices are in the basis vector file. The default file format is the one described in `Basis vectors`_.


cross-validation_folds
~~~~~~~~~~~~~~~~~~~~~~

Variable containing indices of holdout data points, one row per hold-out set. This can be used to define folds for cross-validation or, more generally, hold-out sets for repeated hold-out. The default file format is the one described in `Fold file`_.


model
~~~~~

This variable contains a model learned from a data. It will be generated if user provides a `learner`_ attribute and training data for the learner. Model can be saved into a file via Python's pickle protocol. Previously learned model can be loaded from a file in order to perform predictions for unseen data.


prediction_features
~~~~~~~~~~~~~~~~~~~

Features for data one wishes to make predictions for. Prediction will be performed if a model is loaded from a file or if a predictor has been trained. The default file format is the one described in `Featurefile`_.


test_labels
~~~~~~~~~~~

Correct labels for test data, supply these if you want to measure performance on test data. The default file format is the one described in `Labelfile`_.


predicted_labels
~~~~~~~~~~~~~~~~

Predicted labels for test data. These are generated if a model is used to perform predictions. These are also needed if one wants to measure performance on test data. The default file format is the one described in `Labelfile`_.


prediction_qids
~~~~~~~~~~~~~~~

Qids for test data, supply these if you want to evaluate performance on test data as an average over queries. The default file format is the one described in `Qid file`_.


predicted_clusters_for_training_data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Results of MMC clustering on the training data (see `Clustering with evolutionary maximum margin clustering`_).


selected_features
~~~~~~~~~~~~~~~~~

The indices of the features selected by the GreedyRLS learner (see `Feature selection with greedy RLS`_).


GreedyRLS_LOO_performances
~~~~~~~~~~~~~~~~~~~~~~~~~~

The list containing the LOO performances made by GreedyRLS during the greedy forward selection process (see `Feature selection with greedy RLS`_).

validation_features
~~~~~~~~~~~~~~~~~~~

Variable containing features for validation data. Necessary when using ValidationSetSelection for choosing the regularization parameter. The default file format is the one described in `Featurefile`_.


validation_labels
~~~~~~~~~~~~~~~~~

Variable containing labels for validation data. Necessary when using ValidationSetSelection for choosing the regularization parameter. The default file format is the one described in `Labelfile`_.

validation_qids
~~~~~~~~~~~~~~~

Qids for the validation data. Necessary when using ValidationSetSelection for choosing the regularization parameter, for query structured data with LabelRankRLS, or CGRankRLS (see the `learner`_ attribute).

File formats
============

The following types of files can be supplied as input for rls_core

`Featurefile`_
- the file containing attribute:value pairs for the training
data.

`Labelfile`_
- the file containing the values of the correct labels for training
data.

`Fold file`_
- Indices of holdout data, can be used to define folds for
cross-validation.

`Basis vectors`_
- Indices of the training data points used as basis vectors, for the
reduced set approximation. Normally, all the training data are
basis vectors.

`Qid file`_
- File contains a query id for each data point. This can be
used in query structured ranking tasks to define which document are related to
the same query (information retrieval tasks), to define which
parses correspond to the same sentence (parse ranking), etc.

The convention used when indexing features or data points is to start the
indexing from zero. Thus if there are m distinct features/data points, the
possible indices are from the range [0 ... m-1].

Below we give detailed descriptions of the file formats.

Featurefile
-----------

In all tasks, the data are provided in the input file
one per line using sparse representation.
Technically, the format of a line can be expressed as follows::

    <line> .=. <index>:<value> <index>:<value> ... <index>:<value> # <comment>
    <index> .=. <integer>
    <value> .=. <float>
    <comment> .=. <string>

The features are provided in tokens
consisting of a feature index, a colon, and a real number
indicating the value of the feature. The feature representation is
sparse so that only the features whose values differ
from 0 are present in the line. Further, the feature indices have
to be given from the smallest to the largest starting from zero.
For example, the line::

    0:0.43 3:0.12 9284:0.2

specifies a data point that has non-zero values for features number
0, 3 and 9284, and value 0 for all the other possible features. If
a data point has no non-zero valued attribute, then use ``0:0`` to
differentiate this from empty line.

Labelfile
---------

Labels are the correct output values associated with some set of
data points. These are required in training supervised learners and in
performance estimation, but naturally not when making predictions
for new examples. The labels are provided in the label file so that
each line corresponds to one training data point, the data being
in the same order as in the feature file. The file label file has
the following dense matrix format::

    <line> .=. <value> <value> ... <value> # <comment>
    <value> .=. <float>
    <comment> .=. <string>

Note that there may be several labels per each line but each line
must have the same number of labels. Having multiple labels is
useful for multi-class and multi-label classification tasks or in
general if there are many learning tasks to be solved
simultaneously. For classification 1 is used to represent the
positive class and -1 the negative. For regression and ranking any
real values can be used.

Examples:

Lines::

    1
    -1
    1

Could represent two positive (lines 1 and 3) and one negative
data points in a binary classification task.

Line::

    1 -1 -1 -1 1

could represent the labels for a data point in a multi-label
classification task where a data point may belong to several
different classes simultaneously. In this case the data point would
belong to classes 1 and 5.

Lines::

    1 -1 -1
    -1 -1 1
    -1 -1 1
    -1 1 -1

could represent the labels for four data points in a multi-class
classification task with three possible classes. In this setting
each label corresponds to one class, and each data point has value 1
for the class it belongs to, and -1 for the other classes.

Lines::

    1.123
    3.433
    0.0023

could represent real valued outputs for a simple regression task,
where each data point is associated with one value, which we want to
learn to predict.

Fold file
---------

The cross-validation folds file format is the following. For each
separate hold-out set, there is a line in the file consisting of a
list of indices of the training inputs that belong to the hold-out
set. Technically, the format of a line can be expressed as
follows::

    <line> .=. <index> ... <index> # <comment>
    <index> .=. <integer>
    <comment> .=. <string>

The indices are separated with a white-space character. An index
can not be more than one time in a single line. However, a single
training input can belong to several hold-out sets simultaneously,
and hence an index can be in multiple lines. The indexing of the
training inputs starts from zero.

Basis vectors
-------------

The basis vectors file contains a single line, where the indices of
the basis vectors are contained, separated by whitespace. The
format can be expressed as follows::

    <line> .=. <index> ... <index>
    <index> .=. <integer>

For example::

    0 23 25 44

Would mean that the data points number 0, 23 25 and 44 are used as
basis vectors. An index can not be more than once in this file.
The indexing of the training inputs starts from zero.

Qid file
--------

When performing ranking, the qid value is used to restrict the
pairwise preference relations. By default, the preference relation
covers all pairs of data points. Qids can be used to restrict which
pairs are included in the relation. A pair of data points is included
in the preference relation only, if the value of "qid" is the same
for both of them.

Each line in the query id file contains the id of the query the
data point belongs to. The format can be expressed as follows::

    <line>.=. <qid>
    <qid>.=. <integer>

For example::

    1
    1
    1
    2
    2

Would mean that the first three data points belong to query number 1,
and the last second to query number 2. In this case pairwise
preferences would be observed between the first and second, first
and third, second and third and fourth and fifth data points. However,
preferences between other pairs would not be considered, as they
have different qids. The qids mainly have an effect on the pairwise
performance measures, such as disagreement error or squared
magnitude preserving ranking error. However, they may also have an
effect on the other performance measure due to the averaging over
the queries. For example, if squared error is used together with
the qids provided in the above example file, the average squared
error is first calculated for each query and the overall error is
the average taken over the queries. Therefore, the three first
data points have a lesser weight than the last two data points. This is
in contrast to the case without qids, where the overall error is
the average error taken over all data.

Currently, using qid file and a fold file together is not
supported.


Examples
========

RLScore is designed to be used by calling the appropriate learners from a
python code.

The easiest way to use RLScore is by modifying one of the example python code
files presented next, to match your task. The software supports a wide variety
of different learning tasks, ranging from supervised learning to clustering and
feature selection. Examples of typical use-cases for each type of task are
provided below.

The example code files, and the example data sets used by them can be found
in the 'examples' folder of the RLScore distribution. For example, to run the
example file 'reg_train.py' included in examples/code from the command line,
go to the folder containing the RLScore distribution, and execute the command
'python examples/code/reg_train.py'

While the examples use Unix-style paths with '/' separator,
they work also in Windows with no modifications needed.


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


Python code (classacc_all)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: examples/code/classacc_all.py
   :literal:

    

Binary classification with RankRLS, maximize area under ROC curve (AUC)
-----------------------------------------------------------------------

In binary classification, the data is separated into two classes, which are
often referred to as the positive, and the negative class. AUC measures
the probability, that a randomly drawn positive data point receives a higher
predicted value than a randomly drawn negative one. The measure is especially
suitable for unbalanced data.

When training a classifier according to the AUC criterion, using the
RankRLS learner which minimizes a pairwise least-squares loss on the
training set class labels is recommended. Leave-pair-out cross-validation
is recommended for model selection, unless the data set is very large.


Python code (classAUC_all)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: examples/code/classAUC_all.py
   :literal:



Ranking with RankRLS, minimize pairwise mis-orderings
-----------------------------------------------------

In ranking the aim is to learn a function, whose predictions result in an
accurate ranking when ordering new examples according to the predicted
values. That is, more relevant examples should receive higher predicted
scores than less relevant.

Using qids means that instead of a total order over all examples, each
query has it's own ordering, and examples from different queries should
not be compared. For example in information retrieval, each query
might consist of the ordering of a set of documents according to
a query posed by a user.

When training a ranker, using the RankRLS learner which minimizes a pairwise
least-squares loss on the training set class labels is recommended.
Leave-query-out cross-validation is recommended for parameter selection.

In case you have a total order over all examples, instead of query structrue,
proceed as follows:
- do not supply qid files
- replace LabelRankRLS with AllPairsRankRLS in the Modules section

If the data is both high dimensional and sparse, one should use the module
CGRankRLS, which is optimized for such a data
(see `Learning linear models from large sparse data sets`_).

In addition to learning from utility scores of data points, CGRankRLS also
supports learning from pairwise preferences, see
`Python code (cgrank_test_with_preferences)`_


Python code (rankqids_all)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: examples/code/rankqids_all.py
   :literal:



Regression
----------

In regression, the task is to predict real-valued labels. The regularized
least-squares (RLS) module is suitable for solving this task.


Python code (reg_all)
~~~~~~~~~~~~~~~~~~~~~

.. include:: examples/code/reg_all.py
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


Python code (clustering)
~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: examples/code/clustering.py
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


Python code (fselection)
~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: examples/code/fselection.py
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


Python code (gaussian_kernel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: examples/code/gaussian_kernel.py
   :literal:


Python code (polynomial_kernel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: examples/code/polynomial_kernel.py
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

Python code (cgrls_test)
~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: examples/code/cgrls_test.py
   :literal:


Python code (cgrank_test)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: examples/code/cgrank_test.py
   :literal:


Python code (cgrank_qids)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: examples/code/cgrank_qids.py
   :literal:

Python code (cgrank_test_with_preferences)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: examples/code/cgrank_test_with_preferences.py
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


Python code (reduced_set)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: examples/code/reduced_set.py
   :literal:



Regression with Kronecker kernel
--------------------------------

Regularized least-squares regression with Kronecker kernels is a method
that takes advantage of the computational short-cuts for inverting
so-called shifted Kronecker product systems. The current implementation
only works with the library interface and with kernel matrices for
training and prediction that are constructed in advance.
 

Python code (Kronecker RLS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: examples/code/kron_train_and_predict.py
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







