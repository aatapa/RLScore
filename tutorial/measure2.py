from rlscore.measure import auc

#My class labels, three examples in positive and two in negative
Y = [-1, -1, -1, 1, 1]

#Predict all ties
P = [1, 1, 1, 1, 1]

print("My auc with all ties %f" %auc(Y,P))

#Use Y for prediction
print("My auc with using Y as P is %f" %auc(Y,Y))

#Perfect predictions: AUC is a ranking measure, so all that matters
#is that positive instances get higher predictions than negatives
P2 = [-5, 2, -1, 4, 3.2]

print("My auc with correctly ranked predictions is %f" %auc(Y,P2))

#Let's make the predictions worse

P2 = [-5, 2, -1, 1, 3.2]

print("Now my auc dropped to %f" %auc(Y,P2))

#AUC is undefined if all instances belong to same class, let's crash auc 

Y2 = [1, 1, 1, 1, 1]
#this will not work
auc(Y2, P2)
