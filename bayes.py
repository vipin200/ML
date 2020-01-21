import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
Y = iris.target
print(iris.feature_names)
print(iris.target_names)
# for i in range(len(X)):
#     print(X[i]," ",Y[i], "    ", i)

arr = np.array([])
for i in range(len(X)):
    arr = np.append(arr,X[i][0])
print(arr)
plt.scatter(arr,iris.target, edgecolors='r')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, Y_train)

Y_pred = gnb.predict(X_test)

from sklearn import metrics

print("Accuracy is: ", metrics.accuracy_score(Y_test , Y_pred) * 100)
