import numpy as np

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import qmr

from rlscore.utilities import sampled_kronecker_products
from rlscore.pairwise_predictor import KernelPairwisePredictor
from rlscore.pairwise_predictor import LinearPairwisePredictor


TRAIN_LABELS = 'Y'
CALLBACK_FUNCTION = 'callback'
        
# def dual_objective(a, K1, K2, Y, rowind, colind, lamb):
#     #REPLACE
#     P =  sampled_kronecker_products.sampled_vec_trick(a, K2, K1, rowind, colind, rowind, colind)
#     z = (1. - Y*P)
#     z = np.where(z>0, z, 0)
#     Ka = sampled_kronecker_products.sampled_vec_trick(a, K2, K1, rowind, colind, rowind, colind)
#     return 0.5*(np.dot(z,z)+lamb*np.dot(a, Ka))
# 
# def dual_gradient(a, K1, K2, Y, rowind, colind, lamb):
#     P =  sampled_kronecker_products.sampled_vec_trick(a, K2, K1, rowind, colind, rowind, colind)
#     z = (1. - Y*P)
#     z = np.where(z>0, z, 0)
#     sv = np.nonzero(z)[0]
#     v = P[sv]-Y[sv]
#     v_after = sampled_kronecker_products.sampled_vec_trick(v, K2, K1, rowind, colind, rowind[sv], colind[sv])
#     Ka = sampled_kronecker_products.sampled_vec_trick(a, K2, K1, rowind, colind, rowind, colind)
#     return v_after + lamb*Ka
#         
# def dual_Hu(u, a, K1, K2, Y, rowind, colind, lamb):
#     P =  sampled_kronecker_products.sampled_vec_trick(a, K2, K1, rowind, colind, rowind, colind)            
#     z = (1. - Y*P)
#     z = np.where(z>0, z, 0)
#     sv = np.nonzero(z)[0]
#     v = sampled_kronecker_products.sampled_vec_trick(u, K2, K1, rowind[sv], colind[sv], rowind, colind)
#     v_after = sampled_kronecker_products.sampled_vec_trick(v, K2, K1, rowind, colind, rowind[sv], colind[sv])
#     return v_after + lamb * sampled_kronecker_products.sampled_vec_trick(u, K2, K1, rowind, colind, rowind[sv], colind[sv])


def func(v, X1, X2, Y, rowind, colind, lamb):
    #REPLACE
    #P = np.dot(X,v)
    #P = sampled_kronecker_products.x_gets_subset_of_A_kron_B_times_v(v, X2, X1.T, colind, rowind)
    P = sampled_kronecker_products.sampled_vec_trick(v, X1, X2, colind, rowind)
    z = (1. - Y*P)
    #print z
    z = np.where(z>0, z, 0)
    #return np.dot(z,z)
    return 0.5*(np.dot(z,z)+lamb*np.dot(v,v))

def gradient(v, X1, X2, Y, rowind, colind, lamb):
    #REPLACE
    #P = np.dot(X,v)
    #P = vecProd(X1, X2, v)
    #P = sampled_kronecker_products.x_gets_subset_of_A_kron_B_times_v(v, X2, X1.T, colind, rowind)
    P = sampled_kronecker_products.sampled_vec_trick(v, X1, X2, colind, rowind)
    z = (1. - Y*P)
    z = np.where(z>0, z, 0)
    sv = np.nonzero(z)[0]
    #map to rows and cols
    rows = rowind[sv]
    cols = colind[sv]
    #A = -2*np.dot(X[sv].T, Y[sv])
    #A = - sampled_kronecker_products.x_gets_A_kron_B_times_sparse_v(Y[sv], X1.T, X2, rows, cols)
    A = - sampled_kronecker_products.sampled_vec_trick(Y[sv], X2.T, X1.T, None, None, rows, cols)
    A = A.reshape(X2.shape[1], X1.shape[1]).T.ravel()
    #B = 2 * np.dot(X[sv].T, np.dot(X[sv],v))
    #v_after = sampled_kronecker_products.x_gets_subset_of_A_kron_B_times_v(v, X2, X1.T, cols, rows)
    v_after = sampled_kronecker_products.sampled_vec_trick(v, X1, X2, cols, rows)
    #v_after = sampled_kronecker_products.x_gets_A_kron_B_times_sparse_v(v_after, X1.T, X2, rows, cols)
    v_after = sampled_kronecker_products.sampled_vec_trick(v_after, X2.T, X1.T, None, None, rows, cols)
    B = v_after.reshape(X2.shape[1], X1.shape[1]).T.ravel()
    #print "FOOOBAAR"
    return A + B + lamb*v
    #return -2*np.dot(X[sv].T,Y[sv]) + 2 * np.dot(X[sv].T, np.dot(X[sv],v)) + lamb*v

def hessian(v, p, X1, X2, Y, rowind, colind, lamb):
    #P = np.dot(X,v)
    #P = vecProd(X1, X2, v)
    #P = sampled_kronecker_products.x_gets_subset_of_A_kron_B_times_v(v, X2, X1.T, colind, rowind)
    P = sampled_kronecker_products.sampled_vec_trick(v, X1, X2, colind, rowind)
    z = (1. - Y*P)
    z = np.where(z>0, z, 0)
    sv = np.nonzero(z)[0]
    #map to rows and cols
    rows = rowind[sv]
    cols = colind[sv]
    #p_after = sampled_kronecker_products.x_gets_subset_of_A_kron_B_times_v(p, X2, X1.T, cols, rows)
    p_after = sampled_kronecker_products.sampled_vec_trick(p, X1, X2, cols, rows)
    #p_after = sampled_kronecker_products.x_gets_A_kron_B_times_sparse_v(p_after, X1.T, X2, rows, cols)
    p_after = sampled_kronecker_products.sampled_vec_trick(p_after, X2.T, X1.T, None, None, rows, cols)
    p_after = p_after.reshape(X2.shape[1], X1.shape[1]).T.ravel()
    return p_after + lamb*p    
    #return 2 * np.dot(X[sv].T, np.dot(X[sv],p)) + lamb*p


class KronSVM(object):
        
    
    def __init__(self, **kwargs):
        self.resource_pool = kwargs
        Y = kwargs[TRAIN_LABELS]
        self.label_row_inds = np.array(kwargs["label_row_inds"], dtype = np.int32)
        self.label_col_inds = np.array(kwargs["label_col_inds"], dtype = np.int32)
        self.Y = Y
        self.trained = False
        if kwargs.has_key("regparam"):
            self.regparam = kwargs["regparam"]
        else:
            self.regparam = 1.0
        if kwargs.has_key(CALLBACK_FUNCTION):
            self.callbackfun = kwargs[CALLBACK_FUNCTION]
        else:
            self.callbackfun = None
        if kwargs.has_key("compute_risk"):
            self.compute_risk = kwargs["compute_risk"]
        else:
            self.compute_risk = False
        self.train()
    
    
    def train(self):
        if self.resource_pool.has_key('kmatrix1'):
            self.solve_kernel(self.regparam)
        else:
            self.solve_linear(self.regparam)

    def solve_linear(self, regparam):
        self.regparam = regparam
        X1 = self.resource_pool['xmatrix1']
        X2 = self.resource_pool['xmatrix2']
        self.X1, self.X2 = X1, X2
        
        if 'maxiter' in self.resource_pool: maxiter = int(self.resource_pool['maxiter'])
        else: maxiter = 1000

        if 'inneriter' in self.resource_pool: inneriter = int(self.resource_pool['inneriter'])
        else: inneriter = 50
        
        x1tsize, x1fsize = X1.shape #m, d
        x2tsize, x2fsize = X2.shape #q, r
        lsize = len(self.label_row_inds) #n
        
        kronfcount = x1fsize * x2fsize
        
        label_row_inds = np.array(self.label_row_inds, dtype = np.int32)
        label_col_inds = np.array(self.label_col_inds, dtype = np.int32)
        
        
        #def cgcb(v):
        #    self.W = v.reshape((x1fsize, x2fsize), order = 'F')
        #    self.callback()


        #OPERAATIOT:
        #Z = X1 kron X2 
        #Z * v -> vec-trick: 
        #Z[sv] * v 
        #v_after = sampled_kronecker_products.x_gets_subset_of_A_kron_B_times_v(v, X1, X2.T, label_row_inds, label_col_inds)
        #v_after = sampled_kronecker_products.x_gets_A_kron_B_times_sparse_v(v_after, X1.T, X2, label_row_inds, label_col_inds)
        #Z[sv].T * v
        #

        Y = self.Y
        rowind = label_row_inds
        colind = label_col_inds
        lamb = self.regparam
        rowind = np.array(rowind, dtype = np.int32)
        colind = np.array(colind, dtype = np.int32)
        fdim = X1.shape[1]*X2.shape[1]
        w = np.zeros(fdim)
        #np.random.seed(1)
        #w = np.random.random(fdim)
        self.bestloss = float("inf")
        def mv(v):
            return hessian(w, v, X1, X2, Y, rowind, colind, lamb)
            
        for i in range(maxiter):
            g = gradient(w, X1, X2, Y, rowind, colind, lamb)
            G = LinearOperator((fdim, fdim), matvec=mv, rmatvec=mv, dtype=np.float64)
            self.best_residual = float("inf")
            self.w_new = None
            def cgcb(v):
                residual = np.linalg.norm(G.dot(v)-g)
                #print "residual", residual
                if residual < self.best_residual:
                    self.best_residual = residual
                    self.w_new = v.copy()
            #self.w_new =
            self.w_new = qmr(G, g, maxiter=inneriter)[0]
            #w_new = lsqr(G, g, iter_lim=inneriter)[0]
            #r = G*w_new - g
            #e_rel = np.linalg.norm(r)/np.linalg.norm(g)
            #print e_rel, alpha
            #print "function value", func(w)
            #func(w)
            if np.all(w == w - self.w_new):
                break
            w = w - self.w_new
            if self.compute_risk:
                #P = sampled_kronecker_products.x_gets_subset_of_A_kron_B_times_v(w, X2, X1.T, colind, rowind)
                P = sampled_kronecker_products.sampled_vec_trick(w, X1, X2, colind, rowind)
                z = (1. - Y*P)
                z = np.where(z>0, z, 0)
                loss = 0.5*(np.dot(z,z)+lamb*np.dot(w,w))
                if loss < self.bestloss:
                    self.W = w.reshape((x1fsize, x2fsize), order='C')
                    self.bestloss = loss
            else:
                self.W = w.reshape((x1fsize, x2fsize), order='C')             
            if self.callbackfun != None:
                self.callbackfun.callback(self)
        self.predictor = LinearPairwisePredictor(self.W)

    def solve_kernel(self, regparam):
        self.regparam = regparam
        K1 = self.resource_pool['kmatrix1']
        K2 = self.resource_pool['kmatrix2']
        #X1 = self.resource_pool['xmatrix1']
        #X2 = self.resource_pool['xmatrix2']
        #self.X1, self.X2 = X1, X2
        #x1tsize, x1fsize = X1.shape #m, d
        #x2tsize, x2fsize = X2.shape #q, r
        if 'maxiter' in self.resource_pool: maxiter = int(self.resource_pool['maxiter'])
        else: maxiter = 100
        if 'inneriter' in self.resource_pool: inneriter = int(self.resource_pool['inneriter'])
        else: inneriter = 1000
        lsize = len(self.label_row_inds) #n
        
        label_row_inds = np.array(self.label_row_inds, dtype = np.int32)
        label_col_inds = np.array(self.label_col_inds, dtype = np.int32)
        
        Y = self.Y
        rowind = label_row_inds
        colind = label_col_inds
        lamb = self.regparam
        rowind = np.array(rowind, dtype = np.int32)
        colind = np.array(colind, dtype = np.int32)
        ddim = len(rowind)
        a = np.zeros(ddim)
        self.bestloss = float("inf")
        def func(a):
            #REPLACE
            #P = np.dot(X,v)
            P =  sampled_kronecker_products.sampled_vec_trick(a, K2, K1, rowind, colind, rowind, colind)
            z = (1. - Y*P)
            z = np.where(z>0, z, 0)
            Ka = sampled_kronecker_products.sampled_vec_trick(a, K2, K1, rowind, colind, rowind, colind)
            return 0.5*(np.dot(z,z)+lamb*np.dot(a, Ka))
        def mv(v):
            rows = rowind[sv]
            cols = colind[sv]
            p = np.zeros(len(rowind))
            A =  sampled_kronecker_products.sampled_vec_trick(v, K2, K1, rows, cols, rowind, colind)
            p[sv] = A
            return p + lamb * v
        def rv(v):
            rows = rowind[sv]
            cols = colind[sv]
            p = sampled_kronecker_products.sampled_vec_trick(v[sv], K2, K1, rowind, colind, rows, cols)
            return p + lamb * v
        ssize = 1.0
        for i in range(maxiter):
            P =  sampled_kronecker_products.sampled_vec_trick(a, K2, K1, rowind, colind, rowind, colind)
            z = (1. - Y*P)
            z = np.where(z>0, z, 0)
            sv = np.nonzero(z)[0]
            #print "support vectors", len(sv)
            B = np.zeros(P.shape)
            B[sv] = P[sv]-Y[sv]
            B = B + lamb*a
            #solve Ax = B
            A = LinearOperator((ddim, ddim), matvec=mv, rmatvec=rv, dtype=np.float64)
            #a_new = lsqr(A, B, iter_lim=inneriter)[0]
            #def callback(xk):
            #    print " inner iter zero coefficients:", sum(np.isclose(xk, 0. )), "out of", len(xk) 
            #    residual = np.linalg.norm(A.dot(xk)-B)
            #    print "redidual is", residual
            self.best_residual = float("inf")
            self.a_new = None
            def cgcb(v):
                print v          
                #residual = np.linalg.norm(A.dot(v)-B)
                #print "residual", residual
                #if residual < self.best_residual:
                #    self.best_residual = residual
                #    self.a_new = v.copy()
            self.a_new = qmr(A, B, maxiter=inneriter)[0]
            #a_new = qmr(A, B, x0=a, tol=0.01, callback=callback)[0]
            #a_new = lsmr(A, B, maxiter=inneriter)[0]
            #ssize = 1.0
            if np.all(a == a - self.a_new):
                break
            a = a - self.a_new
            if self.compute_risk:
                loss = func(a)
                if loss < self.bestloss:
                    self.A = a
                    self.bestloss = loss
            else:
                self.A = a
            #print "dual objective", func(a), ssize
            #print "dual objective 2", dual_objective(a, K1, K2, Y, rowind, colind, lamb)
            #print "gradient norm", np.linalg.norm(dual_gradient(a, K1, K2, Y, rowind, colind, lamb))
            self.dual_model = KernelPairwisePredictor(a, rowind, colind)
            if self.callbackfun != None:
                self.callbackfun.callback(self)
            #w = sampled_kronecker_products.x_gets_A_kron_B_times_sparse_v(a, X1.T, X2, rowind, colind)
            #P2 = sampled_kronecker_products.x_gets_subset_of_A_kron_B_times_v(self.W.ravel(), X2, X1.T, colind, rowind)
            #z2 = (1. - Y*P)
            #z2 = np.where(z2>0, z2, 0)
            #print np.dot(z2,z2)
            #self.W = w.reshape((x1fsize, x2fsize), order='F')
            #print w
            #print self.W.ravel()
            #assert False
            #p = func(a)
            #print p[-10:]
            #P2 = sampled_kronecker_products.x_gets_subset_of_A_kron_B_times_v(self.W.ravel(), X2, X1.T, colind, rowind)
            #print "predictions 2", P2
            #print "diff", P1-P2
            #z2 = (1. - Y*P)
            #z2 = np.where(z2>0, z2, 0)
            #print np.dot(z2,z2)
            #print i
        #self.predictor = LinearPairwisePredictor(self.W, X1.shape[1], X2.shape[1])
        self.predictor = KernelPairwisePredictor(a, rowind, colind)
        #z = np.where(a!=0, a, 0)
        #sv = np.nonzero(z)[0]
        #self.predictor = KernelPairwisePredictor([sv], rowind[sv], colind[sv])
        if self.callbackfun != None:
            self.callbackfun.finished(self)

# class KernelPairwisePredictor(object):
#     
#     def __init__(self, A, label_row_inds, label_col_inds, kernel = None):
#         """Initializes the dual model
#         @param A: dual coefficient matrix
#         @type A: numpy matrix"""
#         self.A = A
#         self.label_row_inds, self.label_col_inds = label_row_inds, label_col_inds
#         self.kernel = kernel
#     
#     
#     def predict(self, K1pred, K2pred):
#         """Computes predictions for test examples.
# 
#         Parameters
#         ----------
#         K1pred: {array-like, sparse matrix}, shape = [n_samples1, n_basis_functions1]
#             the first part of the test data matrix
#         K2pred: {array-like, sparse matrix}, shape = [n_samples2, n_basis_functions2]
#             the second part of the test data matrix
#         
#         Returns
#         ----------
#         P: array, shape = [n_samples1, n_samples2]
#             predictions
#         """
#         P = sampled_kronecker_products.x_gets_A_kron_B_times_sparse_v(self.A, K1pred, K2pred.T, self.label_row_inds, self.label_col_inds)
#         P = P.reshape((K1pred.shape[0], K2pred.shape[0]), order='F')
#         return P
# 
#     def predictWithKernelMatricesAlt(self, K1pred, K2pred, row_inds = None, col_inds = None):
#         #transposes wrong probably
#         P =  sampled_kronecker_products.sampled_vec_trick(self.A, K2pred, K1pred, np.array(row_inds, dtype=np.int32), np.array(col_inds, dtype=np.int32), self.label_row_inds, self.label_col_inds)
#         return P

# class LinearPairwisePredictor(object):
#     
#     def __init__(self, W, dim1, dim2):
#         """Initializes the linear model
#         @param W: primal coefficient matrix
#         @type W: numpy matrix"""
#         self.W = W
#         self.dim1, self.dim2 = dim1, dim2
#     
#     
#     def predict(self, X1pred, X2pred):
#         """Computes predictions for test examples.
#         
#         Parameters
#         ----------
#         X1pred: {array-like, sparse matrix}, shape = [n_samples1, n_features1]
#             the first part of the test data matrix
#         X2pred: {array-like, sparse matrix}, shape = [n_samples2, n_features2]
#             the second part of the test data matrix
#         
#         Returns
#         ----------
#         P: array, shape = [n_samples1, n_samples2]
#             predictions
#         """
#         P = np.dot(np.dot(X1pred, self.W), X2pred.T)
#         return P
#     
#     
#     def predictWithDataMatricesAlt(self, X1pred, X2pred, row_inds = None, col_inds = None):
#         if row_inds == None:
#             P = np.dot(np.dot(X1pred, self.W), X2pred.T)
#             P = P.ravel()
#             #P = P.reshape(X1pred.shape[0] * X2pred.shape[0], 1, order = 'F')
#         else:
#             P = sampled_kronecker_products.x_gets_subset_of_A_kron_B_times_v(self.W.reshape((self.W.shape[0] * self.W.shape[1], 1), order = 'F'), X1pred, X2pred.T, np.array(row_inds, dtype=np.int32), np.array(col_inds, dtype=np.int32))
#             #P = X1pred * self.W * X2pred.T
#         return P


