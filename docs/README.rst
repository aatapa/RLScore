=======
RLScore
=======


RLScore - regularized least-squares machine learning algorithms package.


:Authors:         `Tapio Pahikkala <http://staff.cs.utu.fi/~aatapa/>`_,
                  `Antti Airola <https://scholar.google.fi/citations?user=5CPOSr0AAAAJ>`_
:Email:           firstname.lastname@utu.fi
:Homepage:        `https://github.com/aatapa/RLScore <https://github.com/aatapa/RLScore>`_
:Version:         1.0
:License:         `The MIT License <LICENCE.TXT>`_
:Date:            December 10. 2015

.. contents::

Overview
========

RLScore is a machine learning software package for regularized kernel methods,
focusing especially on Regularized Least-Squares (RLS) based methods. The main
advantage of the RLS family of methods is that they admit a closed form solution, expressed as a system of linear equations.
This allows deriving highly efficient algorithms for RLS methods, based on matrix
algebraic optimization. Classical results include computational short-cuts for
multi-target learning, regularization parameter selection and leave-one-out
cross-validation. RLScore takes these results further by implementing a wide
variety of additional computational shortcuts for different types of cross-validation
strategies, single- and multi-target feature selection, multi-task and zero-shot
learning with Kronecker kernels, ranking, stochastic hill climbing based
clustering etc. The majority of the implemented methods are such that are not
available in any other software package.


Support for different tasks
===========================


-  Regression and classification
   
   - Regularized least-squares (RLS)
       - multi-target learning
       - selection of regularization parameter
       - leave-one-out cross-validation
       - leave-pair-out cross-validation
       - k-fold cross-validation
   
-  Feature selection for regression and classification

   - Greedy regularized least-squares (Greedy RLS)
       - greedy forward selection; selects features based on leave-one-out error
       - joint feature selection for multi-target problems
       
-  Ranking

   - Regularized least-squares ranking (GlobalRankRLS)
       - minimizes magnitude preserving ranking error
       - multi-target learning
       - selection of regularization parameter
       - leave-pair-out cross-validation
       - k-fold cross-validation

   - Regularized least-squares ranking for query-structured data (QueryRankRLS)
       - minimizes magnitude preserving ranking error, computed for each query separately
       - multi-target learning
       - selection of regularization parameter
       - leave-query-out cross-validation
       
-   Pairwise data and zero-shot learning

   - Kronecker RLS
