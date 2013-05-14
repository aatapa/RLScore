/* Copyright (c) 2010 the authors listed at the following URL, and/or
the authors of referenced articles or incorporated external code:
http://en.literateprograms.org/Red-black_tree_(C)?action=history&offset=20090121005050

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Retrieved from: http://en.literateprograms.org/Red-black_tree_(C)?oldid=16016
*/

#include "c_swapped.h"
#include <assert.h>
#include <stdlib.h>


typedef rbtree_node node;
typedef enum rbtree_node_color color;


static node grandparent(node n);
static node sibling(node n);
static node uncle(node n);
static color node_color(node n);

static node new_node(double key, color node_color, node left, node right);
static void rotate_left(rbtree t, node n);
static void rotate_right(rbtree t, node n);

static void replace_node(rbtree t, node oldn, node newn);
static void insert_case1(rbtree t, node n);
static void insert_case2(rbtree t, node n);
static void insert_case3(rbtree t, node n);
static void insert_case4(rbtree t, node n);
static void insert_case5(rbtree t, node n);

node grandparent(node n) {
    assert (n != NULL);
    assert (n->parent != NULL); /* Not the root node */
    assert (n->parent->parent != NULL); /* Not child of root */
    return n->parent->parent;
}

node sibling(node n) {
    assert (n != NULL);
    assert (n->parent != NULL); /* Root node has no sibling */
    if (n == n->parent->left)
        return n->parent->right;
    else
        return n->parent->left;
}

node uncle(node n) {
    assert (n != NULL);
    assert (n->parent != NULL); /* Root node has no uncle */
    assert (n->parent->parent != NULL); /* Children of root have no uncle */
    return sibling(n->parent);
}


color node_color(node n) {
    return n == NULL ? BLACK : n->color;
}

rbtree rbtree_create(void) {
    rbtree t = malloc(sizeof(struct rbtree_t));
    t->root = NULL;
    return t;
}

node new_node(double key, color node_color, node left, node right) {
    node result = malloc(sizeof(struct rbtree_node_t));
    result->key = key;
    result->color = node_color;
    result->left = left;
    result->right = right;
    result->size = 1;
    result->duplicates = 0;
    if (left  != NULL)  left->parent = result;
    if (right != NULL) right->parent = result;
    result->parent = NULL;
    return result;
}

int rbtree_smaller(rbtree t, double key) {
    int smaller = 0;
    node n = t->root;
    while (n != NULL) {
        if (key == n->key) {
	  if (n->left != NULL) {
	    smaller += n->left->size;
	  }
	  return smaller;
        } 
	else if (key < n->key) {
	  n = n->left;
        } 
	else {
            assert(key > n->key);
	    if (n->left != NULL) {
	      smaller += n->left->size;
	    }
	    smaller += n->duplicates+1;
            n = n->right;
        }
    }
    return smaller;
}

int rbtree_larger(rbtree t, double key) {
    int larger = 0;
    node n = t->root;
    while (n != NULL) {
        if (key == n->key) {
	  if (n->right != NULL) {
	    larger += n->right->size;
	  }
	  return larger;
        } 
	else if (key > n->key) {
	  n = n->right;
        } 
	else {
            assert(key < n->key);
	    if (n->right != NULL) {
	      larger += n->right->size;
	    }
	    larger += n->duplicates+1;
            n = n->left;
        }
    }
    return larger;
}


void rotate_left(rbtree t, node n) {
    node r = n->right;
    replace_node(t, n, r);
    n->right = r->left;
    if (r->left != NULL) {
        r->left->parent = n;
    }
    r->left = n;
    n->parent = r;
    r->size=n->size;
    n->size=1+n->duplicates;
    if (n->left != NULL) {
        n->size += n->left->size;
    }
   if (n->right != NULL) {
        n->size += n->right->size;
    }
}

void rotate_right(rbtree t, node n) {
    node L = n->left;
    replace_node(t, n, L);
    n->left = L->right;
    if (L->right != NULL) {
        L->right->parent = n;
    }
    L->right = n;
    n->parent = L;
    L->size=n->size;
    n->size=1+n->duplicates;
    if (n->left != NULL) {
        n->size += n->left->size;
    }
   if (n->right != NULL) {
        n->size += n->right->size;
    }
}

void replace_node(rbtree t, node oldn, node newn) {
    if (oldn->parent == NULL) {
        t->root = newn;
    } else {
        if (oldn == oldn->parent->left)
            oldn->parent->left = newn;
        else
            oldn->parent->right = newn;
    }
    if (newn != NULL) {
        newn->parent = oldn->parent;
    }
}

void rbtree_insert(rbtree t, double key) {
    node inserted_node = new_node(key, RED, NULL, NULL);
    if (t->root == NULL) {
        t->root = inserted_node;
    } else {
        node n = t->root;
        while (1) {
	  n->size += 1; /*the size of each visited node grows by one. Note that insertion of a duplicate
			key will also increment the size of the node where duplicate is inserted, and
		       all its ancestors.*/
            if (key == n->key) {
		n->duplicates+=1;
                /* inserted_node isn't going to be used, don't leak it */
                free (inserted_node);
                return;
            } else if (key < n->key) {
                if (n->left == NULL) {
                    n->left = inserted_node;
                    break;
                } else {
                    n = n->left;
                }
            } else {
                assert (key > n->key);
                if (n->right == NULL) {
                    n->right = inserted_node;
                    break;
                } else {
                    n = n->right;
                }
            }
        }
        inserted_node->parent = n;
    }
    insert_case1(t, inserted_node);
}

void insert_case1(rbtree t, node n) {
    if (n->parent == NULL)
        n->color = BLACK;
    else
        insert_case2(t, n);
}

void insert_case2(rbtree t, node n) {
    if (node_color(n->parent) == BLACK)
        return; /* Tree is still valid */
    else
        insert_case3(t, n);
}

void insert_case3(rbtree t, node n) {
    if (node_color(uncle(n)) == RED) {
        n->parent->color = BLACK;
        uncle(n)->color = BLACK;
        grandparent(n)->color = RED;
        insert_case1(t, grandparent(n));
    } else {
        insert_case4(t, n);
    }
}

void insert_case4(rbtree t, node n) {
    if (n == n->parent->right && n->parent == grandparent(n)->left) {
        rotate_left(t, n->parent);
        n = n->left;
    } else if (n == n->parent->left && n->parent == grandparent(n)->right) {
        rotate_right(t, n->parent);
        n = n->right;
    }
    insert_case5(t, n);
}

void insert_case5(rbtree t, node n) {
    n->parent->color = BLACK;
    grandparent(n)->color = RED;
    if (n == n->parent->left && n->parent == grandparent(n)->left) {
        rotate_right(t, grandparent(n));
    } else {
        assert (n == n->parent->right && n->parent == grandparent(n)->right);
        rotate_left(t, grandparent(n));
    }
}

void delete_tree(rbtree t)
{
node n;
node n2;
if (t->root != NULL)
    {
    n = t->root;
    while (n != NULL)
        {
        if (n->left != NULL)
            {
            n = n->left;
            }
        else if (n->right != NULL)
            {
            n = n->right;
            }
        else
           {
           n2 = n;
           n = n->parent;
           if (n != NULL && n->left != NULL)
                {
                n->left = NULL;
                }
            else if (n != NULL)
                {
                n->right = NULL;
                }
            free(n2);
            }
        }
    }
free(t); 
}


double swapped_pairs(int len1, double* s, int len2, double* f) {
  int size = len1;
  int ind;
  int i = 0;
  int j = 0;
  int k;
  int l;
  int m;
  double f_i;
  rbtree t = rbtree_create();
  double swpairs = 0.;
  int ties = 0;
  //for(i=0; i<size; i++){
  while(i<size){
    ties = 0;
    m = i;
    //The smallest-largest prediction pass
    f_i = f[i];
    while(j<size && f[j] == f_i){
        j+=1;
    }
    ties = j-i;
    //Now we must take care of the ties
    if(ties > 1){
       for(k=i; k<j-1; k++){
            for(l=k+1; l<j; l++){
                if(s[k] != s[l]){
                    swpairs += 0.5;}
              			}
    			     }
    		  }
    while(m<j){
        swpairs += rbtree_larger(t, s[m]);
        m += 1;}
    while(i<j){
        rbtree_insert(t, s[i]);
        i += 1;}
      
  }
  delete_tree(t);
  return swpairs;  
  }
