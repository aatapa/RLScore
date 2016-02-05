Perormance measures
===================

RLScore implement a variety of performance measures for classification,
regression and ranking.

Let Y and P contain the
true outputs and predicted outputs for some problem. For single-target
learning problems both are one-dimensional lists or arrays of size [n_samples].
For multi-target problems, both are two-dimensional lists or arrays of
size [n_samples, n_targets].

A performance measure is a function measure(Y,P), that returns a floating
point value denoting how well P matches Y. If Y and P have several columns,
typically the performance measure is computed for each column separately and
then averaged. A performance measure has a property iserror, that is used by
the grid search codes to check whether large or small values are better. An
UndefinedPerformance error may be raised, if for some reason the performance
measure is not well defined for the given input.


Tutorial 1: Basic usage
***********************

First, let us consider some basic binary classification measures. These
performance measures assume that Y-values (true class labels) are from
set {-1,1}. P-values (predicted class labels) can be any real values, but
the are mapped with the rule P[i]>0 -> 1 and P[i]<=0 -> -1, before
computing the performance.

This is how one can compute simple binary classification accuracy.

.. literalinclude:: ../tutorial/measure1.py

.. literalinclude:: ../tutorial/measure1.out

Four out of five instances are correctly classified, so classification accuracy is 0.8.
Giving as input Y-values outside {-1, 1} causes an exception to be raised.

Next, we compute the area under ROC curve.

.. literalinclude:: ../tutorial/measure2.py

.. literalinclude:: ../tutorial/measure2.out

Everything works as one would expect, until we pass Y full of ones to auc. UndefinedPerformance
is raised, because AUC is not defined for problems, where only one class is present in the true
class labels.

Finally, we test cindex, a pairwise ranking measure that computes how many of the pairs where
Y[i] > Y[j] also have P[i] > P[j]. The measure is a generalization of the AUC.

.. literalinclude:: ../tutorial/measure3.py

.. literalinclude:: ../tutorial/measure3.out

We also observe, that when given Y and P with multiple columns, the performance measure is computed
separately for each column, and then averaged. This is what happens when using some performance
measure for parameter selection in cross-validation with multi-output prediction problems. The chosen
parameter is the one that leads to best mean performance over all the targets.

Tutorial 2: Multi-class accuracy
********************************

RLScore contains some tools for converting multi-class learning problems to several independent
binary classification problems, and for converting vector valued multi-target predictions back to
multi-class predictions.

.. literalinclude:: ../tutorial/measure4.py

.. literalinclude:: ../tutorial/measure4.out

When doing multi-class learning, one should use the ova_accuracy function for parameter selection and computing
the final performance.

Tutorial 3: Using your own performance measure
**********************************************

It is quite simple to use your own performance measure with the automated grid search tools for fast regularization
parameter selection.

Data set
--------

We consider the classical
`Boston Housing data set <https://archive.ics.uci.edu/ml/datasets/Housing>`_
from the UCI machine learning repository. The data consists of 506 instances,
13 features and 1 output to be predicted.

The data can be loaded from disk and split into a training set of 250, and test
set of 256 instances using the following code.

.. literalinclude:: ../tutorial/housing_data.py

.. literalinclude:: ../tutorial/housing_data.out

By default, LeaveOneOutRLS select the regularization parameter based on mean-squared error.

.. literalinclude:: ../tutorial/regression3.py
 
.. literalinclude:: ../tutorial/regression3.out

If I want to use cindex instead, I do it like this.

.. literalinclude:: ../tutorial/measure5.py
 
.. literalinclude:: ../tutorial/measure5.out

Mean absolute error is (as of writing) not implemented in RLScore. This is how we can implement
it and use it for parameter selection.

.. literalinclude:: ../tutorial/measure6.py
 
.. literalinclude:: ../tutorial/measure6.out

The only special thing to note here is the property mae.iserror=True, that tells RLScore that the function
mae is an error measure, meaning that lower value means better predictions. If this values is set to
False, the model selection search would choose parameters that lead to highest performance.



