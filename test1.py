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

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
theta = np.zeros((2,(X.shape[1]+1)))
print(theta)
for i  in range(1000):
