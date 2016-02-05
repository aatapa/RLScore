import numpy as np

from rlscore.utilities import multiclass
from rlscore.measure import ova_accuracy

Y = [1,1,2,2,3,3]
Y_ova = multiclass.to_one_vs_all(Y)

P_ova = [[1, 0, 0], [1.2,0.5, 0], [0, 1, -1], [1, 1.2, 0.5], [0.2, -1, -1], [0.3, -1, -2]]
acc = ova_accuracy(Y_ova, P_ova)
print("ova-mapped Y")
print(Y_ova)
print("P, class prediction is chosen with argmax")
print(P_ova)
print("Accuracy computed with one-vs-all mapped labels and predictions: %f" %acc)
print("original Y")
print(Y)
print("P mapped to class predictions")
P = multiclass.from_one_vs_all(P_ova)
print(P)
acc = np.mean(Y==P)
print("Accuracy is the same:%f " %acc)
