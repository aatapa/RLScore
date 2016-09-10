#
# The MIT License (MIT)
#
# This file is part of RLScore 
#
# Copyright (c) 2015 Tapio Pahikkala, Antti Airola
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import cython
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as cnp

cdef enum rbtree_node_color:
    RED, BLACK

cdef struct rbtree_node_t:
    double key
    int size
    int duplicates
    rbtree_node_t* left
    rbtree_node_t* right
    rbtree_node_t* parent
    rbtree_node_color color

ctypedef rbtree_node_t node

ctypedef struct rbtree_t:
    node* root

ctypedef rbtree_t rbtree

ctypedef rbtree_node_color color

cdef node* grandparent(node *n):
    return n.parent.parent

cdef node* sibling(node *n):
    if n == n.parent.left:
        return n.parent.right
    else:
        return n.parent.left

cdef node* uncle(node *n):
    return sibling(n.parent)

cdef color node_color(node *n):
    if n == NULL:
        return BLACK
    else:
        return n.color

cdef rbtree* rbtree_create():
    cdef rbtree* t = <rbtree_t*> malloc(sizeof(rbtree_t));
    t.root = NULL
    return t

cdef node* new_node(double key, color node_color, node *left, node *right):
    cdef node* result = <node*> malloc(sizeof(rbtree_node_t))
    result.key = key
    result.color = node_color
    result.left = left
    result.right = right
    result.size = 1
    result.duplicates = 0;
    result.parent = NULL
    return result

cdef int rbtree_larger(rbtree *t, double key):
    cdef int larger = 0
    cdef node* n = t.root;
    while n != NULL:
        if key == n.key:
            if n.right != NULL:
                larger += n.right.size
            return larger
        elif key > n.key:
            n = n.right
        else:
            if n.right != NULL:
                larger += n.right.size
            larger += n.duplicates+1;
            n = n.left
    return larger

cdef void replace_node(rbtree *t, node *oldn, node *newn):
    if oldn.parent == NULL:
        t.root = newn
    else:
        if oldn == oldn.parent.left:
            oldn.parent.left = newn
        else:
            oldn.parent.right = newn
    if newn != NULL:
        newn.parent = oldn.parent

cdef void rotate_left(rbtree *t, node *n):
    cdef node* r = n.right
    replace_node(t, n, r)
    n.right = r.left;
    if r.left != NULL:
        r.left.parent = n
    r.left = n
    n.parent = r
    r.size = n.size
    n.size = 1 + n.duplicates
    if n.left != NULL:
        n.size += n.left.size
    if n.right != NULL:
        n.size += n.right.size

cdef void rotate_right(rbtree *t, node *n):
    cdef node *L = n.left
    replace_node(t, n, L)
    n.left = L.right
    if L.right != NULL:
        L.right.parent = n
    L.right = n
    n.parent = L
    L.size = n.size
    n.size = 1 + n.duplicates
    if n.left != NULL:
        n.size += n.left.size
    if n.right != NULL:
        n.size += n.right.size

cdef void insert_case1(rbtree *t, node *n):
    if n.parent == NULL:
        n.color = BLACK
    else:
        insert_case2(t, n)

cdef void insert_case2(rbtree *t, node *n):
    if node_color(n.parent) == BLACK:
        return
    else:
        insert_case3(t, n)

cdef void insert_case3(rbtree *t, node *n):
    if node_color(uncle(n)) == RED:
        n.parent.color = BLACK
        uncle(n).color = BLACK
        grandparent(n).color = RED
        insert_case1(t, grandparent(n))
    else:
        insert_case4(t, n)


cdef void insert_case4(rbtree *t, node *n):
    if (n == n.parent.right) and (n.parent == grandparent(n).left):
        rotate_left(t, n.parent)
        n = n.left
    elif (n == n.parent.left) and (n.parent == grandparent(n).right):
        rotate_right(t, n.parent)
        n = n.right
    insert_case5(t, n)

cdef void insert_case5(rbtree *t, node *n):
    n.parent.color = BLACK
    grandparent(n).color = RED
    if (n == n.parent.left) and (n.parent == grandparent(n).left):
        rotate_right(t, grandparent(n))
    else:
        rotate_left(t, grandparent(n))

cdef void rbtree_insert(rbtree *t, double key):
    cdef node *inserted_node = new_node(key, RED, NULL, NULL)
    cdef node *n = t.root
    if t.root == NULL:
        t.root = inserted_node
    else:
        while (True):
            n.size += 1 #the size of each visited node grows by one. Note that insertion of a duplicate key will also increment the size of the node where duplicate is inserted, and all its ancestors
            if key == n.key:
                n.duplicates += 1 #inserted_node isn't going to be used, don't leak it
                free (inserted_node)
                return
            elif key < n.key:
                if n.left == NULL:
                    n.left = inserted_node
                    break
                else:
                    n = n.left
            else:
                if n.right == NULL:
                    n.right = inserted_node
                    break
                else:
                    n = n.right
        inserted_node.parent = n
    insert_case1(t, inserted_node)

cdef void delete_tree(rbtree *t):
    cdef node *n
    cdef node *n2
    if t.root != NULL:
        n = t.root
        while n != NULL:
            if n.left != NULL:
                n = n.left
            elif n.right != NULL:
                n = n.right
            else:
                n2 = n
                n = n.parent
                if (n != NULL) and (n.left != NULL):
                    n.left = NULL
                elif n != NULL:
                    n.right = NULL
                free(n2)
    free(t)

cdef double swapped_pairs(int len1, double *s, int len2, double *f):
    cdef int size = len1
    cdef int ind
    cdef int i = 0
    cdef int j = 0
    cdef int k
    cdef int l
    cdef int m
    cdef double f_i
    cdef rbtree *t = rbtree_create()
    cdef double swpairs = 0.
    cdef int ties = 0
    while i < size:
        ties = 0
        m = i
        #The smallest-largest prediction pass
        f_i = f[i]
        while (j<size) and (f[j] == f_i):
            j+=1
        ties = j-i;
        #Now we must take care of the ties
        if ties > 1:
            for k in range(i, j-1):
                for l in range(k+1, j):
                    if s[k] != s[l]:
                        swpairs += 0.5
        while m<j:
            swpairs += rbtree_larger(t, s[m])
            m += 1
        while i<j:
            rbtree_insert(t, s[i])
            i += 1
    delete_tree(t)
    return swpairs

def count_swapped(cnp.ndarray[cnp.double_t,ndim=1] A, cnp.ndarray[cnp.double_t,ndim=1] B):
    I = np.argsort(B)
    A = A[I]
    B = B[I]
    result = swapped_pairs(A.shape[0], <double*> A.data, B.shape[0], <double*> B.data)
    return result



