Prediction
==========

The RLScore learners make predictions via the learner.predict( ) -function. Usually the user
will not have to worry about the internal details of how this is implemented. In this tutorial
we will look under the hood of RLScore to see how the predictors are implemented.

Tutorial 1: Linear predictor
****************************

Data set
--------

We consider the classical
`Boston Housing data set <https://archive.ics.uci.edu/ml/datasets/Housing>`_
from the UCI machine learning repository. The data consists of 506 instances,
13 features and 1 output to be predicted.

The data can be loaded from disk and split into a training set of 250, and test
set of 256 instances using the following code.

.. literalinclude:: src/housing_data.py

.. literalinclude:: src/housing_data.out

The linear predictor is of the form w.T*x + b. The predictor takes as input
the n_samples x features - data matrix, and computes predictions.

The predictor is stored in the "predictor" -variable of the trained learner.
The "w" and "b" coefficients store the coefficients of the learner model.

.. literalinclude:: src/predictor1.py

.. literalinclude:: src/predictor1.out

Here, we have taken the predictor apart. There are 13 w-coefficients each corresponding
to one feature, and additionally a bias feature. 

Now let us do feature selection using the greedy RLS method, and select 5 most useful
features for the data set.

.. literalinclude:: src/predictor2.py

.. literalinclude:: src/predictor2.out

Now, all but 5 of the w-coefficients are set to zero by the feature selection process.

If we would perform multi-target learning (i.e. Y has several columns), "W" would be
[n_features, n_targets]- sized, and "b" of length n_targets.

Tutorial 2: Non-linear predictor
********************************

Non-linear predictors are of the form K_test * A,
where the rows of K_test correspond  to test instances, columns to training instances,
and elements to kernel evaluations.

.. literalinclude:: src/predictor3.py

.. literalinclude:: src/predictor3.out

Now there are as many dual coefficients as training examples.
If we would perform multi-target learning (i.e. Y has several columns), "A" would be
[n_samples, n_targets]- sized.

The predictor automatically
computes the kernel matrix between training and test examples (unless PrecomputedKernel
option is used, in that case the caller must compute it).


Tutorial 3: Reduced set approximation
*************************************

The reduced set approximation restricts the predictor of the form K_test * A,
where the rows of K_test correspond  to test instances, columns to basis vectors,
and elements to kernel evaluations.

.. literalinclude:: src/predictor4.py

.. literalinclude:: src/predictor4.out

Now the predictor needs to construct only 20 rows for the test kernel matrix.
Again, with multi-target learning A would be of dimensions [n_bvectors, n_targets].

Tutorial 4: Pairwise predictors
*******************************

Data set
--------

The pairwise predictors take as input either two data matrices, or two kernel matrices, corresponding to
two sets of instances. By default, the pairwise models compute the predictions for all pairwise combinations
of the two sets of instances in column-major order [(X1[0], X2[0]), (X1[1], X2[0])...]. However, one can
alternatively supply as argument to the predictor two lists of indices, first corresponding to the rows
of X1 / K1, and the second corresponding to rows of X2 / K2. In this case, predictions are computed only
for these pairs.

For these experiments, we need to download from the drug-target binding affinity data sets page for the Davis et al. data the drug-target interaction affinities (Y), drug-drug 2D similarities (X1), and WS normalized target-target similarities (X2). In the following we will use similarity scores directly as features for the linear kernel, since the similarity matrices themselves are not valid positive semi-definite kernel matrices.

We can load the data set as follows:

.. literalinclude:: src/davis_data.py

.. literalinclude:: src/davis_data.out

Linear pairwise model
---------------------

In this example, we compute predictions first for all combinations of rows from
X1_test and X2_test, and then only for three pairs. The number of coefficients
in the linear model is the product of the number of features in X1 and X2.

.. literalinclude:: src/predictor5.py

.. literalinclude:: src/predictor5.out

Kernel pairwise model
---------------------

Next, we compute predictions first for all combinations of rows from
K1_test and L2_test, and then only for three pairs. The number of coefficients
in the dual model is the product of the number of training examples (rows /columns)
in K1_train and K2_train. For sparse data, the number of dual coefficients would
correspond to number of training pairs with known labels.

.. literalinclude:: src/predictor6.py

.. literalinclude:: src/predictor6.out

Saving the predictor
********************

When saving a predictor for future use one can use either pickle, or alternatively save the model
coefficients directly to a text file. Saving the predictor instead of the learner can save disk
space and memory, since many of the learners store internally matrix decompositions etc. that are
needed by the fast cross-validation and regularization algorithms.

