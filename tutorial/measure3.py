import numpy as np

from rlscore.measure import cindex

#Concordance index is a pairwise ranking measure

#Equivalent to AUC for bi-partite ranking problems
Y = [-1, -1, -1, 1, 1]
P = [-5, 2, -1, 1, 3.2]

cind1 = cindex(Y, P)

print("My cindex is %f" %cind1)

#Can handle also real-valued Y-values

Y2 = [-2.2, -1.3, -0.2, 0.5, 1.1]
#Almost correct ranking, but last two inverted
P2 = [-2.7, -1.1, 0.3, 0.6, 0.5]

cind2 = cindex(Y2, P2)

print("My cindex is %f" %cind2)

#Most performance measures take average over the columns for multi-target problems:

Y_big = np.vstack((Y, Y2)).T
P_big = np.vstack((P, P2)).T
print(Y_big)
print(P_big)
print("(cind1+cind2)/2 %f" %((cind1+cind2)/2.))
print("is the same as cindex(Y_big, P_big) %f" %cindex(Y_big, P_big))
