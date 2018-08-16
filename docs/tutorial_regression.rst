Regression and classification
=============================


Tutorial 1: Basic regression
****************************

In this tutorial, we show how to train a regularized
least-squares (RLS) regressor. We use
the classical computational short-cuts [1]_ for fast
regularization and leave-one-out cross-validation. 

Data set
--------

We consider the classical
`Boston Housing data set <https://archive.ics.uci.edu/ml/machine-learning-databases/housing/>`_
from the UCI machine learning repository. The data consists of 506 instances,
13 features and 1 output to be predicted.

The data can be loaded from disk and split into a training set of 250, and test
set of 256 instances using the following code.

.. literalinclude:: src/housing_data.py

.. literalinclude:: src/housing_data.out


Linear regression with default parameters
-----------------------------------------

In this example we train RLS with default parameters (linear kernel, regparam=1.0)
and compute the mean squared error both for leave-one-out cross-validation,
and for the test set.

In order to check that the resulting model is better than a trivial baseline,
we compare the results to a predictor, that always predicts the mean
of training outputs.

.. literalinclude:: src/regression1.py

The resulting output is as follows.
 
.. literalinclude:: src/regression1.out

Clearly the model works much better than the trivial baseline. Still, since we used
default parameters there is no guarantee that the regularization parameter would
be close to optimal value.

Choosing regularization parameter with leave-one-out
----------------------------------------------------

Next, we choose the regularization parameter based on grid search in an exponential
grid. We select the parameter that leads to lowest leave-one-out cross-validation
error (measured with mean square error). Due to the computational short-cuts
implemented the whole procedure is almost as fast as training RLS once (though on this
small data set the running times would yet not be an issue even for brute force algorithms
that require re-training for each tested parameter and round of cross-validation).

.. literalinclude:: src/regression2.py
 
.. literalinclude:: src/regression2.out

Compared to previous case, we were able to slightly lower the error,
though it turns out in this case the default parameter was already close to optimal.
Usually one cannot expect to be so lucky.

For convenience, the procedure of training RLS and simultaneously selecting the
regularization parameter with leave-one-out is implemented in class LeaveOneOutRLS.

.. literalinclude:: src/regression3.py
 
.. literalinclude:: src/regression3.out

Learning nonlinear predictors using kernels
-------------------------------------------

Next we train RLS using a non-linear kernel function. The Gaussian kernel has a single
parameter, the kernel width gamma, that needs to be selected using cross-validation.
We implement a nested loop, where gamma is selected in the outer loop, and regparam
in the inner. The ordering is important, as it allows us to make use of the fast
algorithm for re-training RLS with different regularization parameter values.

.. literalinclude:: src/regression4.py
 
.. literalinclude:: src/regression4.out

Clearly there is a non-linear relationship between the features and the output,
that the Gaussian kernel allows us to model better than the linear kernel. For
simpler implementation, the inner loop could be replaced with training LeaveOneOutRLS
supplying the grid as a parameter, the results and running time are the same.


.. literalinclude:: src/regression5.py
 
.. literalinclude:: src/regression5.out

Using a custom kernel
---------------------

Custom kernel functions are supported via the kernel="PrecomputedKernel" -option, which
allows the user to directly give a precomputed kernel matrix for training (and later
for prediction). Revisiting the first example, we again train a regressor on the Housing
data:

.. literalinclude:: src/regression6.py

.. literalinclude:: src/regression6.out


Tutorial 2: K-fold cross-validation, non i.i.d. data
****************************************************

Next we consider how to implement K-fold cross-validation. RLS implements computational
shortcuts for repeated holdout, that allow implementing different cross-validation
schemes with almost the same computational cost as training RLS only once. This can
be used to implement regular 10-fold or 5-fold cross-validation if one prefers these
to leave-one-out, though in our opinion the main advantage is realized with non i.i.d.
data, where leave-one-out cannot be used reliably. The computational shortcuts are
based on further refinement of the results published in [2]_ and [3]_, the parse
ranking experiments presented next are similar to the experimental setup considered in [2]_.

We consider an application from the field of natural language processing known as
parse ranking. Syntactic parsing refers to the process of analyzing natural language
text according to some formal grammar. Due to ambiguity of natural language
("I shot an elephant in my pyjamas." - just who was in the pyjamas?), a sentence
can have multiple grammatically correct parses. In parse ranking, an automated parser
generates a set of candidate parsers, which need to be scored according to how
well they match the true (human made) parse of the sentence.

This can be modeled as a regression problem, where the data consists of inputs
representing sentence-parse pairs, and outputs that are scores describing the 'goodness'
of the parse for the sentence. The learned predictor should
generalize to making predictions for new sentences. In order to represent this
requirement in the experimental setup, the training and test data split is done
on the level of sentences, so that all parses related to same sentence are put
either in the training or test set.

Thus a main feature of the data is that it is not an i.i.d. sample, rather the data
is sampled in groups of parses, each group corresponding to one sentence. Similar
violation of the i.i.d. assumption is common in many other machine learning
applications. Typical situations where
this type of cross-validation may be required include setting where several instances correspond
to same real-world object (e.g. same experiment performed many times, same person at different
time points), pairwise data (customer-product, drug-target, query-document, parse-sentence),
data corresponding to points in some coordinate system (pixels in image, spatial data, time
series data) etc.

The data set
------------

The parse ranking data set can be downloaded from 
`here <http://staff.cs.utu.fi/~aatapa/data/parse.zip>`_.


First, we load the training set in and examine its properties

.. literalinclude:: src/parse_data.py
 
.. literalinclude:: src/parse_data.out

As is common in natural language applications the data is very high dimensional. In
addition to the data we load a list of sentence ids, denoting to which sentence each
instance is associated with. Finally, with the map\_ids function, based on the ids we map the data to fold indices, where
each fold contains the indices of all training instances associated with a given sentence.
Altogether there are 117 folds each corresponding to a sentence.

Incorrect analysis using leave-one-out
--------------------------------------

As before, we select the regularization parameter using leave-one-out.


.. literalinclude:: src/parse_regression1.py
 
.. literalinclude:: src/parse_regression1.out

It can be seen that something has gone wrong, the test error is almost 50 times higher than
the leave-one-out error! The problem is that in leave-one-out when a parse is left out of 
the training set, all the other parses related to the same sentence are still left in.
This leads to high optimistic bias in leave-one-out.

K-fold cross-validation to the rescue
-------------------------------------

This time, we perform k-fold cross-validation, where each fold contains instances related
to a single sentence (k=117).


.. literalinclude:: src/parse_regression2.py
 
.. literalinclude:: src/parse_regression2.out

This time everything works out better than before, the cross-validation estimate and test
error are much closer. Interestingly, the regularization parameter chosen based on
the more reliable estimate does not work any better than the one chosen with leave-one-out.


K-fold vs. leave-one-out
------------------------

Finally, we plot the difference between the leave-sentence-out k-fold,
leave-one-out and test set errors.

.. literalinclude:: src/parse_regression_plot.py

.. image:: src/parse_plot.png

The moral of the story is, that if your data is not identically distributed, but rather sampled
in groups, this should be taken into account when designing training/test split and/or the
cross-validation folds. Cross-validation using such custom fold-partition can be implemented
efficiently using the hold-out method implemented in the RLS module. 

Note: One issue with this data set is that even though there is signal present in the data, even the
best results are quite bad in terms of MSE. We cannot really accurately predict the
scores as such.


Tutorial 3: Binary classification and Area under ROC curve
**********************************************************

Adult data set
--------------

In this experiment, we build a binary classifier on the 
`Adult data set <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html>`_.
We model this as a regression problem, where +1 encodes the positive class,
and -1 the negative one. This is the standard encoding assumed by the
performance measures within the RLScore package.

.. literalinclude:: src/adult_data.py
 
.. literalinclude:: src/adult_data.out

Binary classification
---------------------

We can train RLS and select the regularization parameter as before, by
simply using (binary) classification accuracy instead of squared error as the
performance measure.

.. literalinclude:: src/classification0.py
 
.. literalinclude:: src/classification0.out

Area under ROC curve (AUC) and cross-validation
-----------------------------------------------

A common approach in machine learning is to measure performance with area under the ROC curve (AUC) 
rather than classification accuracy. Here, we combine the leave-one-out shortcuts, with using AUC
for parameter selection and performance estimation.

.. literalinclude:: src/classification1.py
 
.. literalinclude:: src/classification1.out

However, as shown for example in [4]_, especially for small data sets leave-one-out can have substantial
bias for AUC-estimation. In this experiment, we split the Adult data set to 1000 separate training sets
of 30 samples, and compare the leave-one-out AUC and test set AUC.

.. literalinclude:: src/classification2.py
 
.. literalinclude:: src/classification2.out

As can be seen, there is a systematic negative bias meaning that the leave-one-out AUCs tend to be
smaller than the AUC on the (quite large) test set. The results are similar to those obtained in
the simulated experiments in [4]_. The bias can be removed by performing leave-pair-out cross-validation,
where on each round of cross-validation one positive-negative example pair is left out of training set.
For this purpose, RLScore implements the fast and exact leave-pair-out algorithm first introduced in
[5]_. What follows is a low-level implementation of leave-pair-out, where all such (x_i, x_j) pairs,
where y_i = +1 and y_j = -1 are left out in turn. The leave-pair-out AUC is the relative fraction of
such pairs, where the f(x_i) > f(x_j), with ties assumed to be broken randomly (see [4]_ for further
discussion).

.. literalinclude:: src/classification3.py
 
.. literalinclude:: src/classification3.out

As can be seen, the negative bias is now almost completely eliminated. 

The above code is very low-level and as such unlikely to be easy to understand or of practical use for most users.
However, leave-pair-out AUC can be automatically computed, and also used for model selection using the
equivalent higher level interface. In the following experiment, we train RLS on a sample of 1000 instances from Adult data
for regularization parameter values 2^-5, 1 and 2^5, and select the predictor corresponding
to highest leave-pair-out AUC.

.. literalinclude:: src/classification4.py
 
.. literalinclude:: src/classification4.out

Tutorial 4: Reduced set approximation
*************************************

Once the data set size exceeds a couple of thousands of instances, maintaining the
whole kernel matrix in memory is no longer feasible. Instead of using all the training
data to represent the learned model, one can instead restrict the model to a subset of
the data, resulting in the so-called reduced set approximation (aka 'subset of regressors',
also closely related to Nyström approximation). As a starting point one can randomly
select a couple of hundred of training instances as basis vectors. A lot of research
has been done on more advanced selection strategies.


.. literalinclude:: src/classification5.py
 
.. literalinclude:: src/classification5.out

Tutorial 5: Multi-target learning
*********************************

RLS supports efficient multi-target learning. Instead of a single vector of outputs Y, one
can supply [instances x targets] shape Y-matrix. For multi-target regression, one can simply
insert the values to be regressed as such. For multi-class or multi-label classification,
one should employ a transformation, where Y-matrix contains one column per class. The
class(es) to which an instance belongs to are encoded as +1, while the others are encoded
as -1 (so called one-vs-all transformation, in case of multi-class problems).

All the cross-validation and fast regularization algorithms are compatible with multi-target
learning.

We demonstrate multi-class learning with a simple toy example, utilizing the `Wine data
set <https://archive.ics.uci.edu/ml/datasets/Wine>`_ from the UCI repository 

.. literalinclude:: src/wine_data.py

.. literalinclude:: src/wine_data.out

We implement the training and testing, using two additional helper functions, one which
transforms class labels to one-vs-all encoding, another that computes classification accuracy
for matrices using one-vs-all encoding.

.. literalinclude:: src/classification6.py
 
.. literalinclude:: src/classification6.out

The wine data turns out to be really easy to learn. Similarly, we could implement multi-target
regression, or multi-label classification. RLScore does not currently implement a wide variety
of multi-label classification measures. However, the implemented performance measures (e.g. accuracy,
auc, f-score), will compute micro-averaged performance estimates (i.e. compute the measure for each
column separately and then average), when applied to 2-dimensional Y and P matrices.


References
**********

.. [1] Ryan Rifkin, Ross Lippert. Notes on Regularized Least Squares. Technical Report, MIT, 2007.

.. [2] Tapio Pahikkala, Jorma Boberg, and Tapio Salakoski. Fast n-Fold Cross-Validation for Regularized Least-Squares. Proceedings of the Ninth Scandinavian Conference on Artificial Intelligence, 83-90, Otamedia Oy, 2006.
    
.. [3] Tapio Pahikkala, Hanna Suominen, and Jorma Boberg. Efficient cross-validation for kernelized least-squares regression with sparse basis expansions. Machine Learning, 87(3):381--407, June 2012. 

.. [4] Antti Airola, Tapio Pahikkala, Willem Waegeman, Bernard De Baets, and Tapio Salakoski An experimental comparison of cross-validation techniques for estimating the area under the ROC curve. Computational Statistics & Data Analysis, 55(4):1828-1844, April 2011.

.. [5] Tapio Pahikkala, Antti Airola, Jorma Boberg, and Tapio Salakoski. Exact and efficient leave-pair-out cross-validation for ranking RLS. In Proceedings of the 2nd International and Interdisciplinary Conference on Adaptive Knowledge Representation and Reasoning (AKRR'08), pages 1-8, Espoo, Finland, 2008.
 
