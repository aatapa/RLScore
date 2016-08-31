Sparse large data sets and linear models
========================================

In some application domains it is typical to have data sets with both large sample size and
dimensionality, but small amount of non-zero feature for each individual instance. For example, in document
classification where features correspond to words the number of possible words in a vocabulary
can be very large, even though a single document will only contain a small subset of these
words.

RLScore contains some tools for learning linear models with such data, combining sparse matrix
data structures with conjugate gradient optimization. The idea was presented for RLS optimization
in [1]_, [2]_ and [3]_ explore the idea further in context of RankRLS training.

While useful tools, these modules are not the main focus of RLScore development,
as the conjugate gradient based training methods are not compatible with the computational
short cuts implemented in the other modules, such as the fast cross-validation algorithms.
For Big Data problems with millions of samples or more and huge dimensionality, we recommend
using stochastic gradient descent solvers available in several other machine learning
libraries.

Data set
--------

We consider the classical
`20 Newsgroups data set <http://qwone.com/~jason/20Newsgroups/>`_. We extract
the preprocessed files in sparse matrix format contained in the archive
file 20news-bydate-matlab.tgz. The data can be processed using the following
code.

.. literalinclude:: src/newsgroups_data.py

.. literalinclude:: src/newsgroups_data.out

Experiments
-----------

In the following, we train RLS using the conjugate gradient algorithm. Unlike
the basic RLS module, CGRLS does not support multi-output data. In this example
we train a classifier for a binary classification task of predicting whether
a document belongs to Newsgroup number 1, or not. Multi-class classification
could be implemented by training one one-vs-all predictor for each class,
and assigning each test instance to the class with the highest prediction.

.. literalinclude:: src/sparse1.py

.. literalinclude:: src/sparse1.out

And the same for RankRLS, with similar results. The CGRankRLS learner supports
also query-structured data, and the learner PCGRankRLS can be used for learning
from pairwise preferences with sparse data.

.. literalinclude:: src/sparse2.py

.. literalinclude:: src/sparse2.out

References
**********

.. [1] Rifkin, R., Yeo, G., & Poggio, T. (2003). Regularized least-squares classification. In J. Suykens, G. Horvath, S. Basu, C. Micchelli, & J. Vandewalle (Eds.), NATO science series III: computer and system sciences: Vol. 190. Advances in learning theory: methods, model and applications (pp. 131–154).

.. [2] Tapio Pahikkala, Evgeni Tsivtsivadze, Antti Airola, Jouni Jarvinen, and Jorma Boberg. An efficient algorithm for learning to rank from preference graphs. Machine Learning, 75(1):129-165, 2009.

.. [3] Antti Airola, Tapio Pahikkala, and Tapio Salakoski. Large Scale Training Methods for Linear RankRLS. ECML/PKDD-10 Workshop on Preference Learning, 2010.


