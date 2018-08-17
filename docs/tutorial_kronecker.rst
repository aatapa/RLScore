Pairwise (dyadic) data, transfer- and zero-shot learning
========================================================

In this tutorial, we consider learning from pairwise (dyadic) data. We assume that the training data
consist a sequence of paired inputs and correct labels ((u, v), y),
and the task is to learn a function f(u,v) = y, that would given a new paired input correctly predict
the label. The setting appears in a wide variety of settings, such as learning interactions for
drug-target pairs, protein-protein interactions, rankings for query-document pairs, customer-product ratings etc.

Pair-input prediction is often considered under the framework of network inference, where the inputs correspond to vertices
of a graph, pairs (u,v) to directed edges, and labels y to edge weights. Here, one can distinguish between two types of graphs. If the start and end nodes u and v belong to different sets, the problem corresponds to predicting edges in a bipartite network. Typical examples would be drug-target interaction, or customer-product rating prediction. On the other hand, if u and v belong to same set, the problem corresponds to edge prediction in a homogenous network. A typical example would be protein-protein interaction prediction. 

For the bipartite case, four settings are commonly recognized. Let us assume making a prediction for a new paired input (u,v):

A. Both u and v are present in the training set, as parts of different labeled pairs, and the label of the pair (u,v) must be predicted.
B. Pairs containing v are present in the training set, while u is not observed in any training pair, and the label of the pair (u,v) must be predicted.
C. Pairs containing u are present in the training set, while v is not observed in any training pair, and the label of the pair (u,v) must be predicted.
D. Neither u nor v occurs in any training pair, and the label of the pair (u,v) must be predicted.

A is the standard setting of matrix completion, that has been considered especially in the context of recommender systems and matrix factorization methods. B and C are examples of multi-target learning
problems, that can also be solved by using the regular RLS. Setting D can be seen as an example
of the zero-shot learning problem, where we need to generalize from related learning tasks to a new
problem, for which we have no training data. For the homogenous network case, settings B and C become equivalent.
For a more detailed overview of these four settings, as well as analysis and comparison of different Kronecker kernel RLS methods, see [1]_ and [2]_. Terminology varies between the articles, the settings considered in [2]_ can be mapped to this tutorial as follows: I=A, R=B, C=C, B=D (bipartite case); E=A, V=B/C (homogenous case).

We assume that the feature representations of the inputs are given either as two data matrices
X1 and X2, or as two kernel matrices K1 and K2. The feature representation can then be formed
as the Kronecker product of these matrices. The Kronecker RLS (KronRLS) method corresponds to
training a RLS on the Kronecker product data or kernel matrices, but is much faster to train due
to computational shortcuts used [3]_ [4]_. Alternatively, a related variant of the method known as
TwoStepRLS may be used [5]_.  The main advantage of TwoStepRLS is that it implements fast
cross-validation algorithms for Settings A - D, as well as related settings for homogenous networks [2]_.

In the following, we will experiment on two
`drug-target binding affinity data sets <http://staff.cs.utu.fi/~aatapa/data/DrugTarget/>`_.
Let us assume that we have n_drugs
and n_targets drugs and targets in our training data. In the complete data setting, we assume
that the correct labels are known for each drug-target-combination in the training data. That is,
the labels can be represented as a n_drugs x n_targets -sized Y-matrix, that has no missing entries.
In practice even if a small amount of the values are missing, this setting can be realized by imputing the missing
entries for example with row- and/or column means. In the incomplete data setting we assume that
many of the values are unknown. These settings are considered separately, since they lead to different
learning algorithms.

Tutorial 1: KronRLS
*******************

In the first tutorial, we consider learning from complete data and compare settings A-D. The experimental
setup is similar to that of [6]_ (results in Table 2, column KdQ). The main difference is that in order
to keep the examples simple and understandable we implement only a single training / test set split, rather
than using proper nested cross-validation.  

Data set
--------

For these experiments, we need to download
from the `drug-target binding affinity data sets <http://staff.cs.utu.fi/~aatapa/data/DrugTarget/>`_ page
for the Davis et al. data the
`drug-target interaction affinities (Y) <http://staff.cs.utu.fi/~aatapa/data/DrugTarget/drug-target_interaction_affinities_Kd__Davis_et_al.2011.txt>`_,
`drug-drug 2D similarities (X1) <http://staff.cs.utu.fi/~aatapa/data/DrugTarget/drug-drug_similarities_2D.txt>`_,
and
`WS target-target similarities (X2) <http://staff.cs.utu.fi/~aatapa/data/DrugTarget/target-target_similarities_WS.txt>`_.
In the following we will use similarity scores directly as features for the linear kernel, since the similarity matrices
themselves are not valid positive semi-definite kernel matrices.

We can load the data set as follows:

.. literalinclude:: src/davis_data.py

.. literalinclude:: src/davis_data.out

There are 68 drugs, 442 targets and 30056 drug-target pairs. The benefit of using KronRLS or TwoStepRLS is that they operate
only on these small matrices, rather than explicitly forming the Kronecker product matrices whose sizes correspond to the
number of pairs. The functions settingx_split() generate a training / test split compatible with the implied setting
(split on level of drugs, targets or both). 

Setting A
---------

First, we consider setting A, imputing missing values inside the Y-matrix. This is not the most interesting problem for this
data set, since we assume complete training set where there are no missing entries in Y in the first place. Still, we can check the in-sample
leave-one-out predictions: if the label of one (u,v) pair is left out of Y, how well can we predict it from the remaining
data? In practice the fast leave-one-out algorithm could be used to impute missing values for example by first replacing
them with mean values, and then computing more accurate estimates via the leave-one-out.

.. literalinclude:: src/kron_rls1.py

.. literalinclude:: src/kron_rls1.out

The best results are for regparam 2**21, with concordance index (e.g. pairwise ranking accuracy, generalization of AUC
for real-valued data) around 0.90.

Setting B
---------

Next, we consider setting B, generalizing to predictions for new drugs that were not observed in the training set.

.. literalinclude:: src/kron_rls2.py

.. literalinclude:: src/kron_rls2.out

The results show that this problem is really quite different from setting A: the best results are now much lower than in Setting A. Also, the required amount of regularization is quite different, suggesting that selecting regularization parameter with leave-one-out may be a bad idea, if the goal is to generalize to the other settings.

Setting C
---------

Next, we consider setting C, generalizing to predictions for new targets that were not observed in the training set.

.. literalinclude:: src/kron_rls3.py

.. literalinclude:: src/kron_rls3.out

Again quite different results.

Setting D
---------

Finally we consider the most demanding setting D, generalizing to new (u,v) pairs such that neither have been
observed in the training set.

.. literalinclude:: src/kron_rls4.py

.. literalinclude:: src/kron_rls4.out

The results are noticeably lower than for any other setting, yet still much better than the random baseline 0.5 meaning that the model still has predictive power.

Using kernels
-------------

By default KronRLS assumes linear kernel for both input domains. However, KronRLS also supports
the use of pre-computed kernel matrices. In the following, we repeat experiment
for setting D, this time first computing the kernel matrices, before passing them to the learner. One can
use any proper kernel function for learning, and the instances from the first and second domain may
often have different types of kernel functions. Some very basic kernel functions are implemented in the
module rlscore.kernel.


.. literalinclude:: src/kron_rls5.py

.. literalinclude:: src/kron_rls5.out

Results are same as before.


Tutorial 2: TwoStepRLS, cross-validation with bipartite network
***************************************************************

For complete data, another method that can be used in the TwoStepRLS algorithm [5]_.
The algorithm has two different regularization parameters, one that can be used in our
experiments to regularize the drugs, and one the targets. The main advantage of the method
is that it allows for fast cross-validation methods [2]_. In the following experiments, we
consider cross-validations for settings A - D. We use the data set introduced in previous example. 

Setting A
---------

Leave-pair-out cross-validation. On each round of CV, one (drug,target) pair is left out of the training set as test pair. 

.. literalinclude:: src/two_step1.py

.. literalinclude:: src/two_step1.out

Kfold cross-validation. Same as above, but several (drug, target) pairs left out at once.

.. literalinclude:: src/two_step1b.py

.. literalinclude:: src/two_step1b.out

Setting B
---------

Leave-drug-out cross-validation. On each CV round, a single holdout drug is left out, and all (drug, target) pairs
this drug belongs to used as the test fold.

.. literalinclude:: src/two_step2.py

.. literalinclude:: src/two_step2.out

Kfold cross-validation. Same as above, but several drugs left out at once.

.. literalinclude:: src/two_step2b.py

.. literalinclude:: src/two_step2b.out


Setting C
---------

Leave-target-out cross-validation. On each CV round, a single holdout target is left out, and all (drug, target) pairs
this target belongs to used as the test fold.

.. literalinclude:: src/two_step3.py

.. literalinclude:: src/two_step3.out

Kfold cross-validation. Same as above, but several targets left out at once.

.. literalinclude:: src/two_step3b.py

.. literalinclude:: src/two_step3b.out


Setting D
---------

Out-of-sample leave-pair-out. On each CV round, a single (drug, target) pair is used as test pair (similar to setting A). However, all
pairs where either the drug or the target appears, are left out of the training set.

.. literalinclude:: src/two_step4.py

.. literalinclude:: src/two_step4.out

Kfold cross-validation. Same as above, but several (drug, target) pairs left out at once.

.. literalinclude:: src/two_step4b.py

.. literalinclude:: src/two_step4b.out

Tutorial 2: TwoStepRLS, cross-validation with homogenous network
***************************************************************

Here, our goal is to predict one type of drug similarity matrix (ECFP4 similarities) from another (2D similarities). For these experiments, we need to download
from the `drug-target binding affinity data sets <http://staff.cs.utu.fi/~aatapa/data/DrugTarget/>`_ page
for the Metz et al. data the
`drug-drug ECFP4 similarities (Y) <http://staff.cs.utu.fi/~aatapa/data/DrugTarget/drug-drug_similarities_ECFP4.txt>`_ and
`drug-drug 2D similarities (X) <http://staff.cs.utu.fi/~aatapa/data/DrugTarget/drug-drug_similarities_2D__Metz_et_al.2011.txt>`_.

These implementations are a work in progress, and the inteface may still change. Currently, only the kernel versions are implemented, and only symmetric (f(u,v) = f(v,u)) or antisymmetric (f(u,v) = -f(v,u)) labels are supported.

Setting A
---------

Leave-vertex-out cross-validation. On each round of CV, a single (drug_i, drug_j) pair is left out of the training set as test pair. 

.. literalinclude:: src/two_step_symmetric1.py

.. literalinclude:: src/two_step_symmetric1.out

Setting B/C
-----------

Leave-edge-out cross-validation. On each CV round, a single holdout drug d_i is left out, and all (drug_i, drug_j) pairs
this drug belongs to are used as the test fold. 

.. literalinclude:: src/two_step_symmetric2.py

.. literalinclude:: src/two_step_symmetric2.out

Setting D
---------

Out-of-sample leave-one-out. On each CV round, a single (drug_i, drug_j) pair is used as test pair (similar to setting A). However, all other pairs where either of these drugs appears, are left out of the training set.

.. literalinclude:: src/two_step_symmetric3.py

.. literalinclude:: src/two_step_symmetric3.out

Tutorial 3: CGKronRLS, incomplete data
***************************

In many applications one does not have available the correct labels for all (u,v) pairs
in the training set. Rather, only a (possibly small) fraction of these are known. Next,
we consider learning from such data, using an iterative Kronecker RLS training algorithm,
based on a generalization of the classical Vec trick shortcut for Kronecker products, and
a conjugate gradient type of optimization
approach [7]_. CGKronRLS is an iterative training algorithm, in the next experiments
we will use a callback function to check, how the test set predictive accuracy behaves
as a function of training iterations. Early stopping of optimization has a regularizing
effect, which is seen to be beneficial especially in setting D.

Data set
--------

For these experiments, we need to download
from the `drug-target binding affinity data sets <http://staff.cs.utu.fi/~aatapa/data/DrugTarget/>`_ page
for the Metz et al. data the
`drug-target interaction affinities (Y) <http://staff.cs.utu.fi/~aatapa/data/DrugTarget/known_drug-target_interaction_affinities_pKi__Metz_et_al.2011.txt>`_,
`drug-drug 2D similarities (X1) <http://staff.cs.utu.fi/~aatapa/data/DrugTarget/drug-drug_similarities_2D__Metz_et_al.2011.txt>`_,
and
`WS normalized target-target similarities (X2) <http://staff.cs.utu.fi/~aatapa/data/DrugTarget/target-target_similarities_WS_normalized__Metz_et_al.2011.txt>`_.
In the following we will use similarity scores directly as features for the linear kernel, since the similarity matrices
themselves are not valid positive semi-definite kernel matrices.

We can load the data set as follows:

.. literalinclude:: src/metz_data.py

.. literalinclude:: src/metz_data.out

The code includes four functions to split the data into a training and test set according to the four different settings.

Setting A
---------

First, we consider setting A, imputing missing values inside the Y-matrix. 

.. literalinclude:: src/cg_kron_rls1.py

.. literalinclude:: src/cg_kron_rls1.out

Even at 1000 iterations the test set cindex is still slowly increasing, it might be beneficial to continue optimization
even further.

Setting B
---------

Next, we consider setting B, generalizing to predictions for new drugs that were not observed in the training set.

.. literalinclude:: src/cg_kron_rls2.py

.. literalinclude:: src/cg_kron_rls2.out

Now the performance peaks around hundred iterations, and starts slowly decreasing. In this setting,
regularization by early stopping seems to be beneficial.

Setting C
---------

Next, we consider setting C, generalizing to predictions for new targets that were not observed in the training set.

.. literalinclude:: src/cg_kron_rls3.py

.. literalinclude:: src/cg_kron_rls3.out

Behaviour is similar as in setting A, the performance is still slowly increasing when optimization is terminated.

Setting D
---------

Finally we consider the most demanding setting D, generalizing to new (u,v) pairs such that neither have been
observed in the training set.

.. literalinclude:: src/cg_kron_rls4.py

.. literalinclude:: src/cg_kron_rls4.out

Now the best results are reached soon after 100 iterations, after which the model starts to overfit.


References
**********

.. [1] Michiel Stock, Tapio Pahikkala, Antti Airola, Bernard De Baets, and Willem Waegeman. A Comparative Study of Pairwise Learning Methods Based on Kernel Ridge Regression. Neural Computation, 30(8):2245--2283, 2018.
.. [2] Michiel Stock, Tapio Pahikkala, Antti Airola, Willem Waegeman, and Bernard De Baets. (2018). Algebraic shortcuts for leave-one-out cross-validation in supervised network inference. bioRxiv, 242321.
.. [3] Tapio Pahikkala, Willem Waegeman, Antti Airola, Tapio Salakoski, and Bernard De Baets. Conditional ranking on relational data. In José L. Balcázar, Francesco Bonchi, Aristides Gionis, and Michèle Sebag, editors, Machine Learning and Knowledge Discovery in Databases (ECML PKDD 2010), volume 6322 of Lecture Notes in Computer Science, pages 499--514. Springer, 2010.
.. [4] 	Tapio Pahikkala, Antti Airola, Michiel Stock, Bernard De Baets, and Willem Waegeman. Efficient regularized least-squares algorithms for conditional ranking on relational data. Machine Learning, 93(2-3):321--356, 2013.
.. [5] Tapio Pahikkala, Michiel Stock, Antti Airola, Tero Aittokallio, Bernard De Baets, and Willem Waegeman. A two-step learning approach for solving full and almost full cold start problems in dyadic prediction. In Toon Calders, Floriana Esposito, Eyke Hüllermeier, and Rosa Meo, editors, Machine Learning and Knowledge Discovery in Databases (ECML PKDD 2014), volume 8725 of Lecture Notes in Computer Science, pages 517--532. Springer, 2014.
.. [6] Tapio Pahikkala, Antti Airola, Sami Pietilä, Sushil Shakyawar, Agnieszka Szwajda, Jing Tang, and Tero Aittokallio. Toward more realistic drug-target interaction predictions. Briefings in Bioinformatics, 16(2):325--337, 2015.
.. [7] Antti Airola, and Tapio Pahikkala. "Fast Kronecker Product Kernel Methods via Generalized Vec Trick." IEEE transactions on neural networks and learning systems 99 (2017): 1-14.

