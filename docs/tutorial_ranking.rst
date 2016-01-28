Learning to rank
=============================


In this tutorial, we show how to train the ranking regularized least-squares (RankRLS)
method for learning to rank [1]_ [2]_. We will use three variants of the method, depending
on whether the data consists of (instance, utility score) pairs similar to
regression, query-structured data, or pairwise preferences. In our experience
for the first case
competitive results can often be achieved also by simply using RLS regression,
whereas for the latter two use cases RankRLS should be used. All of these learners
support using nonlinear kernels.

RankRLS minimizes the magnitude preserving ranking error ((y_i-y_j) - (f(x_i) - f(x_j)))^2.
We will also make use of the concordance index (a.k.a pairwise ranking accuracy), that computes
the relative fraction of correctly ordered pairs (s.t. y_i > y_j and f(x_i) > f(x_j) with tied
predictions broken randomly). For concordance index, trivial baselines such as random predictor
or mean or majority voter yield 0.5 performance. For bipartite ranking tasks where there are only two possible output values, the
concordance index is equivalent to area under ROC curve (AUC), a popular measure in binary
classification.

Tutorial 1: Ordinal regression
******************************

First, let us assume an ordinal regression type of setting, where similar to
regression each instance is associated with a score. However, now the aim is
to learn to predict the ordering of instances correctly, rather than the
scores exactly. We use the GlobalRankRLS implementation of the RankRLS.
Global in the name refers to the fact that there exists a single global
ranking over the data, rather than having many separate rankings such
as with query structured data considered later.

The leave-pair-out cross-validation approach consists of leaving in turn
each pair of training instances out as holdout data, and computing the
fraction of cases where f(x_i) > f(x_j), assuming y_i > y_j. This is
implemented using a fast algorithm described in [3]_ [2]_.

Data set
--------

Again, we consider the classical
`Boston Housing data set <https://archive.ics.uci.edu/ml/datasets/Housing>`_
from the UCI machine learning repository. The data consists of 506 instances,
13 features and 1 output to be predicted.

The data can be loaded from disk and split into a training set of 250, and test
set of 256 instances using the following code.

.. literalinclude:: ../tutorial/housing_data.py

.. literalinclude:: ../tutorial/housing_data.out

Linear ranking model with default parameters
--------------------------------------------

First, we train RankRLS with default parameters (linear kernel, regparam=1.0)
and compute the concordance for the test set.

.. literalinclude:: ../tutorial/ranking1.py

The resulting output is as follows.
 
.. literalinclude:: ../tutorial/ranking1.out

Clearly the model works much better than the trivial baseline or 0.5.


Leave-pair-out cross-validation
-------------------------------

Next, we use the fast leave-pair-out cross-validation algorithm for performance
estimation and regularization parameter selection. The LeavePairOutRankRLS module
is a high level interace to this functionality.

.. literalinclude:: ../tutorial/ranking2.py

The resulting output is as follows.
 
.. literalinclude:: ../tutorial/ranking2.out

We notice two things. First, the leave-pair-out estimates are very close to 
the cindex computed on the test set. Second, on this data set the ranking
accuracy does not seem to be much affected by the choice of regularization
parameter.

K-fold cross-validation
-----------------------

Even with the computational shortcuts, leave-pair-out cross-validation
becomes impractical and unnecessary once training set size grows beyond
couple of hundreds of instances. Instead, fast holdout method may be
used to compute k-fold cross-validation estimates for RankRLS as follows.

.. literalinclude:: ../tutorial/ranking3.py

The resulting output is as follows.
 
.. literalinclude:: ../tutorial/ranking3.out

Again, we may also use higher level wrapper code, that can be also used
to select regularization parameter.


.. literalinclude:: ../tutorial/ranking4.py

The resulting output is as follows.
 
.. literalinclude:: ../tutorial/ranking4.out

Kernel parameter selection
--------------------------

Finally, we consider how to select together regularization parameter and
kernel parameters using k-fold cross-validation (alternatively, leave-pair-out
could also be used here).

.. literalinclude:: ../tutorial/ranking5.py

The resulting output is as follows.
 
.. literalinclude:: ../tutorial/ranking5.out

The results are quite similar as before, the Gaussian kernel does not
allow us to outperform the linear one.

Tutorial 2: Query-structured data
*********************************

Next we consider the setting, where instead of having a singe global ranking, the data
is partitioned into subsets ("queries"). Each instance corresponds to a query-object pair,
and the pairs corresponding to the same query have a ranking defined between them.
At test time the model needs to predict rankings for new queries. The terminology comes from the
the context of learning to rank for information retrieval, where the task is given a user
query to rank documents, according to how well they match the query. Terms such as listwise
ranking, conditional ranking and dyadic ranking have also been used, and the setting is
similar to that of label ranking.

We consider an application from the field of natural language processing known as
parse ranking. Syntactic parsing refers to the process of analyzing natural language
text according to some formal grammar. Due to ambiguity of natural language
("I shot an elephant in my pyjamas." - just who was in the pyjamas?), a sentence
can have multiple grammatically correct parses. In parse ranking, an automated parser
generates a set of candidate parsers, which need to be scored according to how
well they match the true (human made) parse of the sentence.

This can be modeled as a ranking problem, where the data consists of inputs
representing sentence-parse pairs, and outputs that are scores describing the 'goodness'
of the parse for the sentence. Each sentence corresponds to a query, and the parses
of that sentence are objects that should be ranked.

QueryRankRLS minimizes the magnitude preserving ranking error within each training
query, but does not compare instances belonging to different queries.

The data set
------------

First, we load the training set in and examine its properties

.. literalinclude:: ../tutorial/parse_data.py
 
.. literalinclude:: ../tutorial/parse_data.out

As is common in natural language applications the data is very high dimensional. In
addition to the data we load a list of sentence ids, denoting to which sentence each
instance belongs to. Finally, based on the ids we map the data to fold indices, where
each fold contains the indices of all training instances associated with a given sentence.
Altogether there are 117 folds each corresponding to a sentence, the
ids for two first folds are printed on screen.

Learning a ranking function
---------------------------

First, we learn a ranking function. The main difference to learning global rankings is
how the queries are handled. When computing the cindex for test set, we compute the
cindex for each test query separately, and finally take the mean.

.. literalinclude:: ../tutorial/parse_ranking1.py
 
.. literalinclude:: ../tutorial/parse_ranking1.out

Leave-query-out cross-validation
------------------------------

Next, we estimate the ranking accuracy on training set using leave-query-out cross-validation,
where each query is in turn left out of the training set as holdout set. This is implemented
using the efficient algorithm described in [2]_.

.. literalinclude:: ../tutorial/parse_ranking2.py
 
.. literalinclude:: ../tutorial/parse_ranking2.out

Leave-query-out and model selection
-----------------------------------

Finally, we use a higher leve wrapper for the leave-query-out, and also make use of automated
model selection at the same time.

.. literalinclude:: ../tutorial/parse_ranking3.py
 
.. literalinclude:: ../tutorial/parse_ranking3.out

Tutorial 3: Learning from pairwise preferences
**********************************************

RankRLS also supports training data given as pairwise preferences of the type A > B meaning that
instance A is preferred to instance B. In the next example, we generate a set of such pairwise
preferences sampled from the Housing data training set, train a model and test to see how well
the model predicts on independent test data.

.. literalinclude:: ../tutorial/pairwise_ranking.py
 
.. literalinclude:: ../tutorial/pairwise_ranking.out

With a sample of 1000 training pairs, the model works as well as the one trained on the y-scores
directly. 

Precomputed kernels, reduced set approximation
**********************************************

See the regression tutorial for examples. RankRLS also supports precomputed kernel matrices
supplied together with the kernel="PrecomputedKernel" argument, as well as reduced set
approximation for large data sets.


References
**********
    
.. [1] Tapio Pahikkala, Evgeni Tsivtsivadze, Antti Airola, Jorma Boberg and Tapio Salakoski Learning to rank with pairwise regularized least-squares. In Thorsten Joachims, Hang Li, Tie-Yan Liu, and ChengXiang Zhai, editors, SIGIR 2007 Workshop on Learning to Rank for Information Retrieval, pages 27--33, 2007.
    
.. [2] Tapio Pahikkala, Evgeni Tsivtsivadze, Antti Airola, Jouni Jarvinen, and Jorma Boberg. An efficient algorithm for learning to rank from preference graphs. Machine Learning, 75(1):129-165, 2009.

.. [3] Tapio Pahikkala, Antti Airola, Jorma Boberg, and Tapio Salakoski. Exact and efficient leave-pair-out cross-validation for ranking RLS. In Proceedings of the 2nd International and Interdisciplinary Conference on Adaptive Knowledge Representation and Reasoning (AKRR'08), pages 1-8, Espoo, Finland, 2008.
 
