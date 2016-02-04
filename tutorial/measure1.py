from rlscore.measure import accuracy

#My class labels, three examples in positive and two in negative
Y = [-1, -1, -1, 1, 1]

#Some predictions
P = [-1, -1, 1, 1, 1]

print("My accuracy %f" %accuracy(Y,P))

#Accuracy accepts real-valued predictions, P2[i]>0 are mapped to +1, rest to -1
P2 = [-2.7, -1.3, 0.2, 1.3, 1]

print("My accuracy with real-valued predictions %f" %accuracy(Y,P2))

Y2 = [2, 1 , 3, 4, 1]

#Labels must be in the set {-1,1}, this will not work

accuracy(Y2, P)
