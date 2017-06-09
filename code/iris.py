#!/bin/python
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns
sns.set_context('talk')

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names
feature_names = iris.feature_names

indices_0 = [i for i, x in enumerate(y) if x == 0]
indices_1 = [i for i, x in enumerate(y) if x == 1]
indices_2 = [i for i, x in enumerate(y) if x == 2]

xx = 2
yy = 3

plt.plot(X[indices_0, xx], X[indices_0, yy], 'o')
plt.plot(X[indices_1, xx], X[indices_1, yy], 'o')
plt.plot(X[indices_2, xx], X[indices_2, yy], 'o')

plt.xlabel(feature_names[xx], fontsize=30)
plt.ylabel(feature_names[yy], fontsize=30)
plt.ylim([min(X[:,yy])-0.1*max(X[:,yy]), max(X[:,yy])*1.1])
plt.xlim([min(X[:,xx])-0.1*max(X[:,xx]), max(X[:,xx])*1.1])
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()
plt.savefig('iris_1.eps', format='eps', dpi=1000)
plt.show()

plt.plot(X[indices_1, xx], X[indices_1, yy], 'o')
plt.plot(X[indices_2, xx], X[indices_2, yy], 'o')
plt.plot(X[indices_0, xx], X[indices_0, yy], 'o')

plt.xlabel(feature_names[xx], fontsize=30)
plt.ylabel(feature_names[yy], fontsize=30)
plt.ylim([min(X[:,yy])-0.1*max(X[:,yy]), max(X[:,yy])*1.1])
plt.xlim([min(X[:,xx])-0.1*max(X[:,xx]), max(X[:,xx])*1.1])
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()
plt.savefig('iris_2.eps', format='eps', dpi=1000)
plt.show()
