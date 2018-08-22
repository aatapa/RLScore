Performance measures
====================

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

.. literalinclude:: src/measure1.py

.. literalinclude:: src/measure1.out

Four out of five instances are correctly classified, so classification accuracy is 0.8.
Giving as input Y-values outside {-1, 1} causes an exception to be raised.

Next, we compute the area under ROC curve.

.. literalinclude:: src/measure2.py

.. literalinclude:: src/measure2.out

Everything works as one would expect, until we pass Y full of ones to auc. UndefinedPerformance
is raised, because AUC is not defined for problems, where only one class is present in the true
class labels.

Finally, we test cindex, a pairwise ranking measure that computes how many of the pairs where
Y[i] > Y[j] also have P[i] > P[j]. The measure is a generalization of the AUC.

.. literalinclude:: src/measure3.py

.. literalinclude:: src/measure3.out

We also observe, that when given Y and P with multiple columns, the performance measure is computed
separately for each column, and then averaged. This is what happens when using some performance
measure for parameter selection in cross-validation with multi-output prediction problems. The chosen
parameter is the one that leads to best mean performance over all the targets.

Tutorial 2: Multi-class accuracy
********************************

RLScore contains some tools for converting multi-class learning problems to several independent
binary classification problems, and for converting vector valued multi-target predictions back to
multi-class predictions.

.. literalinclude:: src/measure4.py

.. literalinclude:: src/measure4.out

When doing multi-class learning, one should use the ova_accuracy function for parameter selection and computing
the final performance.



