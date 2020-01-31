from sklearn.datasets import load_iris
from sklearn.utils import shuffle
import numpy as np
import random
import math
def train_test_split(iris , test_size):
    X = iris.data
    Y = iris.target
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    X = X[randomize]
    Y = Y[randomize]
    X_train , X_test = np.split(X, [int((1-test_size)*len(X))])
    Y_train , Y_test = np.split(Y, [int((1-test_size)*len(Y))])
    return X_train ,X_test ,Y_train,Y_test

iris = load_iris()
X_train,X_test,Y_train,Y_test = train_test_split(iris , 0.4)

no_attr = len(X_train[0])     # no of attributes
un_class ,count_class = np.unique(Y_train , return_counts=True)    # get all unique classes
no_class = len(un_class)
ii = np.empty(no_class ,dtype=np.ndarray)
for i in range(no_class):
    ii[i] = np.where(Y_train == un_class[i])
xi = np.empty((no_class,no_attr) , dtype=np.ndarray)
# print(X_train[ii[0]])

print()
print()
# print(temp1)
for i in range(no_class):
    for j in range(no_attr):
        temp = X_train[ii[i]]
        xi[i][j] = temp[:,j]

# print(xi[0][0])

mean_xi = np.empty((no_class,no_attr))
std_xi = np.empty_like(mean_xi)
for i in range(no_class):
    for j in range(no_attr):
        mean_xi[i][j] = np.mean(xi[i][j])
        std_xi[i][j] = np.std(xi[i][j])

# print(mean_xi)
prob_class = count_class/sum(count_class)

prob_yx = np.empty((len(X_test),no_class))

def prob_xy(X , i, j):  # i : class  -------- j: attribute
    temp = math.exp(-math.pow(X-mean_xi[i][j],2) / (2 * math.pow(std_xi[i][j],2)))
    temp1 = (1/math.sqrt(2 * math.pi * std_xi[i][j])) * temp
    return temp1

for i in range(len(X_test)):

    for j in range(no_class):
        prob_yx[i][j] = prob_class[j]
        for k in range(no_attr):
            prob_yx[i][j] *= prob_xy(X_test[i][k] , j , k)



# print(prob_yx)
max_prob = np.argmax(prob_yx , axis=1)
# print(max_prob)
# print(Y_test)
check = max_prob == Y_test
# print(check)

true_count = np.count_nonzero(check)
# print(true_count)

accuracy = (true_count/len(Y_test)) *100
print("Accuracy calculated: ",accuracy)





from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, Y_train)

Y_pred = gnb.predict(X_test)

from sklearn import metrics

print("Accuracy is: ", metrics.accuracy_score(Y_test , Y_pred) * 100)
