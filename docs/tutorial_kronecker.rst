Pairwise (dyadic) data, transfer- and zero-shot learning
========================================================

In this tutorial, we consider learning from pairwise (dyadic) data. We assume that the training data
consist a sequence ((d, t), y) of paired inputs and correct outputs,
and the task is to learn a function f(d,t) = y, that would given a new paired input correctly predict
the output. The setting appears in a wide variety of settings, such as learning interactions for
drug-target pairs, rankings for query-document pairs, customer-product ratings etc.

Four settings are commonly recognized. Let us assume making a prediction for a new paired input (d,t):

1. Both t and d are observed during training, as parts of separate inputs.
2. d is known during training, while t is not observed
3. t is known during training, while d is not observed
4. Neither t nor d occur in any training input

1 is the standard setting of matrix completion, that has been considered especially in the context of
recommender systems and matrix factorization methods. 2 and 3 are examples of multi-target learning
problems, that can also be solved by using the regular RLS. The focus of this tutorial is on setting
4, while the considered methods can also be used in the other settings. This can be seen as an example
of the zero-shot learning problem, where we need to generalize from related learning tasks to a new
problem, for which we have no training data.

We assume that the feature representations of the inputs are given either as two data matrices
X1 and X2, or as two kernel matrices K1 and K2. The feature representation can then be formed
as the Kronecker product of these matrices. The Kronecker RLS (KronRLS) method corresponds to
training a RLS on the Kronecker product data or kernel matrices, but is much faster to train due
to computational shortcuts used [1]_ [2]_. 

Alternatively we can use a two-step training approach. Here, in the first phase a multi-target RLS predictor
g( ) is trained using X1 and Y. Next, a second RLS predictor is trained from X2 and g(X1), that is, using
the predictions of the first phase as training labels [3]_ We call this approach TwoStepRLS. It can be
shown that KronRLS and TwoStepRLS are very closely related and give very similar results in practice. 
However TwoStepRLS is more flexible as it allows separate regularization for the two input domains, and
better cross-validation approaches.

In the following, we will experiment on two
`drug-target binding affinity data sets <http://staff.cs.utu.fi/~aatapa/data/DrugTarget/>`_.
Let us assume that we have n_drugs
and n_targets drugs and targets in our training data. In the complete data - setting, we assume
that the correct outputs are known for each drug-target-combination in the training data. That is,
the outputs can be represented as a n_drugs x n_targets -sized Y-matrix, that has no missing entries.
In practice even if a small amount of the values are missing, this setting can be realized by imputing the missing
entries for example with row- and/or column means. In the incomplete data setting we assume that
many of the values are unknown. These settings are considered separately, since they lead to different
learning algorithms.

Tutorial 1: Settings 1-4, KronRLS, complete data
************************************************

In the first tutorial, we consider learning from complete data and compare settings 1-4. The experimental
setup is similar to that of [4]_ (results in Table 2, column KdQ). The main difference is that in order
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

Setting 1
---------

First, we consider setting 1, imputing missing values inside the Y-matrix. This is not the most interesting problem for this
data set, since we assume complete training set where there are no missing entries in Y in the first place. Still, we can check the in-sample
leave-one-out predictions: if the label of one (d,t) pair is left out of Y, how well can we predict it from the remaining
data? In practice the fast leave-one-out algorithm could be used to impute missing values for example by first replacing
them with mean values, and then computing more accurate estimates via the leave-one-out.

.. literalinclude:: src/kron_rls1.py

.. literalinclude:: src/kron_rls1.out

The best results are for regparam 2**21, with concordance index (e.g. pairwise ranking accuracy, generalization of AUC
for real-valued data) around 0.90.

Setting 2
---------

Next, we consider setting 2, generalizing to predictions for new drugs that were not observed in the training set.

.. literalinclude:: src/kron_rls2.py

.. literalinclude:: src/kron_rls2.out

The results show that this problem is really quite different from setting 1: the best results are now around
0.75 c-index, a significant drop from setting 1. Also, the required amount of regularization is quite different
(best results fro 2**29), suggesting that selecting regularization parameter with leave-one-out may be
a bad idea, if the goal is to generalize to the other settings.

Setting 3
---------

Next, we consider setting 3, generalizing to predictions for new targets that were not observed in the training set.

.. literalinclude:: src/kron_rls3.py

.. literalinclude:: src/kron_rls3.out

Again quite different results. Best results 0.87 c-index aroung regparam 2**22.

Setting 4
---------

Finally we consider the most demanding setting 4, generalizing to new (d,t,) pairs such that neither have been
observed in the training set.

.. literalinclude:: src/kron_rls4.py

.. literalinclude:: src/kron_rls4.out

Now the best results are around c-index 0.71, with regparam 2**29. The results are noticeably lower than for
any other setting, yet still much better than the random baseline 0.5 meaning that the model still has
predictive power.

Using kernels
-------------

By default KronRLS assumes linear kernel for both input domains. However, KronRLS also supports
the use of pre-computed kernel matrices. In the following, we repeat experiment
for setting 4, this time first computing the kernel matrices, before passing them to the learner. One can
use any proper kernel function for learning, and the instances from the first and second domain may
often have different types of kernel functions. Some very basic kernel functions are implemented in the
module rlscore.kernel.


.. literalinclude:: src/kron_rls5.py

.. literalinclude:: src/kron_rls5.out

Results are same as before.


Tutorial 2: Settings 1-4, TwoStepRLS, complete data
************************************************

For complete data, another method that can be used in the TwoStepRLS algorithm [3]_.
The algorithm has two different regularization parameters, one that can be used in our
experiments to regularize the drugs, and one the targets. The following experiments
are similar to those of Kronecker RLS, but now we also make use of the efficient
cross-validation algorithms for TwoStepRLS.

Setting 1
---------


.. literalinclude:: src/two_step1.py

.. literalinclude:: src/two_step1.out



Setting 2
---------


.. literalinclude:: src/two_step2.py

.. literalinclude:: src/two_step2.out


Setting 3
---------

.. literalinclude:: src/two_step3.py

.. literalinclude:: src/two_step3.out


Setting 4
---------


.. literalinclude:: src/two_step4.py

.. literalinclude:: src/two_step4.out


Tutorial 3: Incomplete data
***************************

In many applications one does not have available the correct labels for all (d,t) pairs
in the training set. Rather, only a (possibly small) fraction of these are known. Next,
we consider learning from such data, using an iterative Kronecker RLS training algorithm,
based on fast sampled Kronecker products and conjugate gradient type of optimization
approach [5]_. CGKronRLS is an iterative training algorithm, in the next experiments
we will use a callback function to check, how the test set predictive accuracy behaves
as a function of training iterations. Early stopping of optimization has a regularizing
effect, which is seen to be beneficial especially in setting 4.

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

Setting 1
---------

First, we consider setting 1, imputing missing values inside the Y-matrix. 

.. literalinclude:: src/cg_kron_rls1.py

.. literalinclude:: src/cg_kron_rls1.out

Even at 1000 iterations the test set cindex is still slowly increasing, it might be beneficial to continue optimization
even further.

Setting 2
---------

Next, we consider setting 2, generalizing to predictions for new drugs that were not observed in the training set.

.. literalinclude:: src/cg_kron_rls2.py

.. literalinclude:: src/cg_kron_rls2.out

Now the performance peaks between hundred and two hundred iterations, and starts slowly decreasing. In this setting,
regularization by early stopping seems to be beneficial.

Setting 3
---------

Next, we consider setting 3, generalizing to predictions for new targets that were not observed in the training set.

.. literalinclude:: src/cg_kron_rls3.py

.. literalinclude:: src/cg_kron_rls3.out

Behaviour is similar as in setting 1, the performance is still slowly increasing when optimization is terminated.

Setting 4
---------

Finally we consider the most demanding setting 4, generalizing to new (d,t,) pairs such that neither have been
observed in the training set.

.. literalinclude:: src/cg_kron_rls4.py

.. literalinclude:: src/cg_kron_rls4.out

Now the best results are reached soon after 100 iterations, after which the model starts to overfit.


References
**********

.. [1] Tapio Pahikkala, Willem Waegeman, Antti Airola, Tapio Salakoski, and Bernard De Baets. Conditional ranking on relational data. In José L. Balcázar, Francesco Bonchi, Aristides Gionis, and Michèle Sebag, editors, Machine Learning and Knowledge Discovery in Databases (ECML PKDD 2010), volume 6322 of Lecture Notes in Computer Science, pages 499--514. Springer, 2010.
.. [2] 	Tapio Pahikkala, Antti Airola, Michiel Stock, Bernard De Baets, and Willem Waegeman. Efficient regularized least-squares algorithms for conditional ranking on relational data. Machine Learning, 93(2-3):321--356, 2013.
.. [3] Tapio Pahikkala, Michiel Stock, Antti Airola, Tero Aittokallio, Bernard De Baets, and Willem Waegeman. A two-step learning approach for solving full and almost full cold start problems in dyadic prediction. In Toon Calders, Floriana Esposito, Eyke Hüllermeier, and Rosa Meo, editors, Machine Learning and Knowledge Discovery in Databases (ECML PKDD 2014), volume 8725 of Lecture Notes in Computer Science, pages 517--532. Springer, 2014.
.. [4] Tapio Pahikkala, Antti Airola, Sami Pietilä, Sushil Shakyawar, Agnieszka Szwajda, Jing Tang, and Tero Aittokallio. Toward more realistic drug-target interaction predictions. Briefings in Bioinformatics, 16(2):325--337, 2015.
.. [5] Tapio Pahikkala. Fast gradient computation for learning with tensor product kernels and sparse training labels. In Pasi Fränti, Gavin Brown, Marco Loog, Francisco Escolano, and Marcello Pelillo, editors, Structural, Syntactic, and Statistical Pattern Recognition (S+SSPR 2014), volume 8621 of Lecture Notes in Computer Science, pages 123--132. Springer, 2014.

