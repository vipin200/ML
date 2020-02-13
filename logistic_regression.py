import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

def normalize(X):
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    return X

dataset = load_breast_cancer()
X = dataset.data
Y = dataset.target

for i in range(X.shape[1]):
    X[:,i] = normalize(X[:,i])
X = np.concatenate((np.ones((len(X),1)),X),axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
theta = np.zeros(X.shape[1])                           # theta ----> 1 x n
J = np.empty(20000)
m = len(X_train)
alpha = 0.05
for i  in range(20000):
    z = X_train @ theta.T
    y_pred = 1/(1+np.exp(-z))
    J[i] = (-1/m) * ((Y_train.T @ np.log(y_pred)) + ((1-Y_train).T @ np.log(1-y_pred)))
    theta = theta - (alpha/m) * (X_train.T @ (y_pred-Y_train))

plt.plot(np.arange(20000),J)
plt.show()
y_pred_test = X_test @ theta.T
y_pred_test[y_pred_test >= 0.5] = 1
y_pred_test[y_pred_test < 0.5] = 0
check = y_pred_test == Y_test
true_count = np.count_nonzero(check)
accuracy = true_count/len(Y_test) *100
print(accuracy)