from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
Y = iris.target

# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
iris = sns.load_dataset('iris')
g = sns.PairGrid(iris, hue = 'species')
g.map(plt.scatter)
g.add_legend()

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=1)

k = range(1, 21)
accuracy = []

for i in k:
    from sklearn.neighbors import KNeighborsClassifier

    knc = KNeighborsClassifier(i)
    knc.fit(X_train, Y_train)

    Y_pred = knc.predict(X_test)

    from sklearn.metrics import accuracy_score

    accuracy.append(accuracy_score(Y_test, Y_pred) * 100)

    from sklearn.metrics import confusion_matrix

    print(confusion_matrix(Y_test, Y_pred))
    print()

plt.scatter(k, accuracy)
plt.xlabel('k')
plt.ylabel('Accuracy(in %)')
plt.show()

