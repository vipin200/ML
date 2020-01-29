from sklearn.datasets import load_iris
from sklearn.utils import shuffle
import numpy as np
import random
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
un_class = np.unique(Y_train)    # get all unique classes
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

print(mean_xi)