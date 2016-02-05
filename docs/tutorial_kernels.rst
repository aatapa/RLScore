Kernels
=======

Most of the algorithms in RLScore are kernel methods. The default behavior of
these methods is to use the linear kernel, if no additional arguments are
provided. However, many of the learners allow supplying as parameters kernel
name and parameters and/or support pre-computed kernel matrices supplied by
a user. If the number of features is small, training with linear kernel
can be much faster, than with the non-linear ones. Also the non-linear kernels are
compatible with the fast cross-validation and regularization algorithms introduced
in the tutorials.

Some further examples on using kernels can be found in the other tutorials.

RLScore currently implements the following kernels:

LinearKernel: k(xi,xj) = <xi , xj> + bias
parameters: bias (default = 1.0)

GaussianKernel: k(xi,xj) = e^(-gamma*<xi-xj,xi-xj>)
paramaters: gamma (default = 1.0)

PolynomialKernel: k(xi,xj) = (gamma * <xi, xj> + coef0)**degree
parameters: gamma (default = 1.0), coef0 (default = 0.), degree (default = 2)

PrecomputedKernel: caller will supply their own kernel matrix for training and prediction, instead of the data matrix.

Tutorial 1: Basic usage
***********************

In these examples we use the RLS learner (see regression and classification tutorials).
The RankRLS modules (see learning to rank tutorials) behave exactly the same with respect
to use of kernel functions.

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


Default behaviour
-----------------------------------------

By default, if no kernel related parameters are supplied, the
learners use linear kernel with bias=1.0. 


.. literalinclude:: ../tutorial/regression1.py

The resulting output is as follows.
 
.. literalinclude:: ../tutorial/regression1.out

Different kernels
-----------------------------------------

Alternatively, we can use the kernels implemented in RLScore. 

First, let us try Gaussian kernel with the default parameters.

.. literalinclude:: ../tutorial/kernel4.py
 
.. literalinclude:: ../tutorial/kernel4.out

Oops, the results are horribly bad. Clearly, the default parameters are far from optimal.

Next, we cheat a bit by using prior knowledge about good kernel parameter values. 
In the other tutorials there are examples on how the fast cross-validation algorithms
can be used to automatically select the regularization and kernel parameters.

Linear kernel:

.. literalinclude:: ../tutorial/kernel1.py
 
.. literalinclude:: ../tutorial/kernel1.out

Gaussian kernel:

.. literalinclude:: ../tutorial/kernel2.py
 
.. literalinclude:: ../tutorial/kernel2.out

Polynomial kernel:

.. literalinclude:: ../tutorial/kernel3.py
 
.. literalinclude:: ../tutorial/kernel3.out

Reduced set approximation
-----------------------------------------

Once the data set size exceeds a couple of thousands of instances, maintaining the whole kernel matrix in memory is no longer feasible. Instead of using all the training data to represent the learned model, one can instead restrict the model to a subset of the data, resulting in the so-called reduced set approximation (aka ‘subset of regressors’, also closely related to Nyström approximation). As a starting point one can randomly select a couple of hundred of training instances as basis vectors. A lot of research has been done on more advanced selection strategies.

.. literalinclude:: ../tutorial/kernel5.py
 
.. literalinclude:: ../tutorial/kernel5.out

Compared to previous example on using Gaussian kernel, the accuracy of the model slightly degrades due to approximation. Training is now faster and uses less memory (though you will not notice the difference yet on this data), and restricting the model to 100 basis vectors instead of all 250 training examples makes prediction roughly 2.5 time faster.

Tutorial 2 Precomputed kernels
******************************

With the "PrecomputedKernel" option, you can supply precomputed kernel matrices instead of data matrix as input to the learner. 

Basic usage
-----------

In the basic case, one needs to supply a n_samples x n_samples -sized (valid positive semi-definite) kernel matrix for training.
For prediction, we compute a n_test_samples x n_samples -sized matrix containing kernel evaluations between test and training data.
In this example, we generate the kernel matrices using the Gaussian kernel implementation in RLScore.

.. literalinclude:: ../tutorial/kernel6.py
 
.. literalinclude:: ../tutorial/kernel6.out

Reduced set approximation
-------------------------

Precomputed kernel can also be combined with subset of regressors approximation. In this case, use "PrecomputedKernel"-option. For training
supply the n_samples x n_bvectors -slice of full kernel matrix instead of data matrix, and n_bvectors x n_bvectors -slice of kernel matrix
instead of basis vectors. In testing, supply n_test_samples x n_samples -sized matrix containing kernel evaluations between test examples
and basis vectors.

.. literalinclude:: ../tutorial/kernel7.py
 
.. literalinclude:: ../tutorial/kernel7.out

The results are exactly the same, as with previous subset of regressors example.

Tutorial 3 Kronecker learners
*****************************

The Kronecker kernel type of learners take as input either two data matrices, or two kernel matrices. The final (only implicitly formed)
data / kernel matrix is the Kronecker product of these two matrices.

For these experiments, we need to download from the drug-target binding affinity data sets page for the Davis et al. data the drug-target interaction affinities (Y), drug-drug 2D similarities (X1), and WS normalized target-target similarities (X2). In the following we will use similarity scores directly as features for the linear kernel, since the similarity matrices themselves are not valid positive semi-definite kernel matrices.

We can load the data set as follows:

.. literalinclude:: ../tutorial/davis_data.py

.. literalinclude:: ../tutorial/davis_data.out

Linear Kernel
-------------------------

Default behaviour when supplying two data matrices is to use linear kernel for both domains:

.. literalinclude:: ../tutorial/kron_rls4.py

.. literalinclude:: ../tutorial/kron_rls4.out

Precomputed kernels
-------------------------

Alternatively, pre-computed kernel matrices may be supplied. The two kernels for the different
domains need not be the same.

.. literalinclude:: ../tutorial/kron_rls6.py

.. literalinclude:: ../tutorial/kron_rls6.out
