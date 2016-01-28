Feature selection
=================

In this section we consider the Greedy RLS algorithm, that performs feature selection
in order to learn sparse linear models. Conceptually the method combines together a greedy search
heuristic for feature selection, regularized least-squares based model fitting, and 
leave-one-out cross-validation based selection criterion. However, unlike an implementation based
on writing wrapper code around a RLS solver, Greedy RLS uses several computational shortcuts to
guarantee that the feature selection can be done in linear time. On multi-target data, the method
selects such common features that work on average best for all the target tasks.

Greedy RLS was first introduced in [1]_. Work on applying and adapting the method to analyzing
genome wide association studies can be found in [2]_, while the multi-target feature selection problem has
been considered in detail in [3]_.

In the
literature one can usually find three main motivations for feature selection:

1. costs: features cost time and money, so let's use as few as possible
2. accuracy: use only the relevant features in order to make more accurate predictions
3. understandability: we can examine the model directly to gain insight into the studied phenomenon

In our opinion, Greedy RLS is especially beneficial for the point 1: the method will search for
such a subset that allows as accurate predictions as possible given a hard constraint on the
number of features allowed. For point 2 we recommend using the regularization mechanisms of the
regular RLS instead. For point 3 we note that analyzing the selected features can be quite challenging,
due to the greedy search process applied. For example, an important feature may end up not being
selected, if it has a high correlation with already selected features.

Basics of feature selection
***************************

Again, we test the algorithm on a toy example, the
`Boston Housing data set <https://archive.ics.uci.edu/ml/datasets/Housing>`_
from the UCI machine learning repository. The data consists of 506 instances,
13 features and 1 output to be predicted.

The data can be loaded from disk and split into a training set of 250, and test
set of 256 instances using the following code.

.. literalinclude:: ../tutorial/housing_data.py

.. literalinclude:: ../tutorial/housing_data.out

Selecting k features
--------------------

First, we test selecting 5 features.

.. literalinclude:: ../tutorial/fselection1.py

.. literalinclude:: ../tutorial/fselection1.out

The feature indices range from 0 to fcount-1.

Callback function
-----------------

GreedyRLS takes as an optional argument a callback function. This is an object that implements two
methods: callback(learner) that is called each time a feature is selected, and finished(learner) that
is called after the selection process has ended. The callback function can be used for example to
follow how test set prediciton error evolves as a function of selected features.

.. literalinclude:: ../tutorial/fselection2.py

.. literalinclude:: ../tutorial/fselection2.out

Running GreedyRLS on all the features produces a ranking of the features: the most important,
the second most important given the already selected one, etc. The final model is exactly the
same as regular RLS trained on all the features.


Feature selection with MNIST
****************************

In this tutorial, we select features for the `MNIST handwritten digit recognition data set
<http://yann.lecun.com/exdb/mnist/>`_. For reading in the data, we use the
`python-mnist package <https://pypi.python.org/pypi/python-mnist/>`_. (Easiest way to install: 'pip install python-mnist').

There are now 60000 training examples, 10000 test examples, 784 features and 10 classes. The goal is to
select such common features that allow jointly predicting all the 10 classes as well as possible. The
classes are not linearly separable, with no pre-processing done the best a linear classifier can achieve
with all the features is around 0.88 accuracy.

.. literalinclude:: ../tutorial/greedy_mnist.py

.. literalinclude:: ../tutorial/greedy_mnist.out

On this data using 50 selected features we achieve classification accuracy of 0.82, better accuracy could
still be gained by continuing the selection process further.

References
**********

.. [1] Tapio Pahikkala, Antti Airola, and Tapio Salakoski. Speeding up Greedy Forward Selection for Regularized Least-Squares. Proceedings of The Ninth International Conference on Machine Learning and Applications, 325-330, IEEE Computer Society, 2010.

.. [2] Tapio Pahikkala, Sebastian Okser, Antti Airola, Tapio Salakoski, and Tero Aittokallio. Wrapper-based selection of genetic features in genome-wide association studies through fast matrix operations. Algorithms for Molecular Biology, 7(1):11, 2012.

.. [3] Pekka Naula, Antti Airola, Tapio Salakoski, and Tapio Pahikkala. Multi-label learning under feature extraction budgets. Pattern Recognition Letters, 40:56--65, April 2014. 
