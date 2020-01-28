from sklearn.datasets import load_iris
from sklearn.utils import shuffle
import numpy as np
import random
def train_test_split(iris , test_size):
    X = iris.data
    Y = iris.target
    print(type(iris))
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    X = X[randomize]
    Y = Y[randomize]
    X_train , X_test = np.split(X, [int((1-test_size)*len(X))])
    Y_train , Y_test = np.split(Y, [int((1-test_size)*len(X))])
    return X_train ,X_test ,Y_train,Y_test

iris = load_iris()
X_train,X_test,Y_train,Y_test = train_test_split(iris , 0.4)

