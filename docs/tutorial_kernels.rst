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
parameters: gamma (default = 1.0)

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

.. literalinclude:: src/housing_data.py

.. literalinclude:: src/housing_data.out


Default behavior
-----------------------------------------

By default, if no kernel related parameters are supplied, the
learners use linear kernel with bias=1.0. 


.. literalinclude:: src/regression1.py

The resulting output is as follows.
 
.. literalinclude:: src/regression1.out

Different kernels
-----------------------------------------

Alternatively, we can use the kernels implemented in RLScore. 

First, let us try Gaussian kernel with the default parameters.

.. literalinclude:: src/kernel4.py
 
.. literalinclude:: src/kernel4.out

Oops, the results are horribly bad. Clearly, the default parameters are far from optimal.

Next, we cheat a bit by using prior knowledge about good kernel parameter values. 
In the other tutorials there are examples on how the fast cross-validation algorithms
can be used to automatically select the regularization and kernel parameters.

Linear kernel:

.. literalinclude:: src/kernel1.py
 
.. literalinclude:: src/kernel1.out

Gaussian kernel:

.. literalinclude:: src/kernel2.py
 
.. literalinclude:: src/kernel2.out

Polynomial kernel:

.. literalinclude:: src/kernel3.py
 
.. literalinclude:: src/kernel3.out

Reduced set approximation
-----------------------------------------

Once the data set size exceeds a couple of thousands of instances, maintaining the whole kernel matrix in memory is no longer feasible. Instead of using all the training data to represent the learned model, one can instead restrict the model to a subset of the data, resulting in the so-called reduced set approximation (aka ‘subset of regressors’, also closely related to Nyström approximation). As a starting point one can randomly select a couple of hundred of training instances as basis vectors. A lot of research has been done on more advanced selection strategies.

.. literalinclude:: src/kernel5.py
 
.. literalinclude:: src/kernel5.out

Compared to previous example on using Gaussian kernel, the accuracy of the model slightly degrades due to approximation. Training is now faster and uses less memory (though you will not notice the difference yet on this data), and restricting the model to 100 basis vectors instead of all 250 training examples makes prediction roughly 2.5 time faster.

Tutorial 2 Precomputed kernels
******************************

With the "PrecomputedKernel" option, you can supply precomputed kernel matrices instead of data matrix as input to the learner. 

Basic usage
-----------

In the basic case, one needs to supply a n_samples x n_samples -sized (valid positive semi-definite) kernel matrix for training.
For prediction, we compute a n_test_samples x n_samples -sized matrix containing kernel evaluations between test and training data.
In this example, we generate the kernel matrices using the Gaussian kernel implementation in RLScore. Note that one first initializes a GaussianKernel object with a set of training data. All subsequent calls of the getKM method return a kernel matrix consisting of all kernel evaluations between the training data and a set of data points provided as an argument. For example, the call getKM(X\_test) provides a kernel matrix between the training and test data.

.. literalinclude:: src/kernel6.py
 
.. literalinclude:: src/kernel6.out

Reduced set approximation
-------------------------

Precomputed kernel can also be combined with reduced set approximation. In this case, use "PrecomputedKernel"-option. For training
supply the n_samples x n_bvectors -slice of full kernel matrix instead of data matrix, and n_bvectors x n_bvectors -slice of kernel matrix
instead of basis vectors. In testing, supply n_test_samples x n_bvectors -sized matrix containing kernel evaluations between test examples
and basis vectors.

.. literalinclude:: src/kernel7.py
 
.. literalinclude:: src/kernel7.out

The results are exactly the same, as with previous reduced set example.

Tutorial 3 Kronecker learners
*****************************

The Kronecker kernel type of learners take as input either two data matrices, or two kernel matrices. The final (only implicitly formed)
data / kernel matrix is the Kronecker product of these two matrices.

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

Linear Kernel
-------------------------

Default behavior when supplying two data matrices is to use linear kernel for both domains:

.. literalinclude:: src/kron_rls4.py

.. literalinclude:: src/kron_rls4.out

Precomputed kernels
-------------------------

Alternatively, pre-computed kernel matrices may be supplied. The two kernels for the different
domains need not be the same.

.. literalinclude:: src/kron_rls6.py

.. literalinclude:: src/kron_rls6.out
