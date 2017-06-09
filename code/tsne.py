#!/bin/python
# coding=utf-8
#
# Creates test subsets of the large original DBs for testing
#
# Author: Javier Sagastuy
# Stand: 19.04.2017
#

import os, itertools, random, progressbar, csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets

fname = '/Users/jvrsgsty/Documents/ITAM/Tesis/Batches/BatchT4_JSB/data/encoded_DBs/mc_Quijote1_500_200E_800G.dat'
data = np.loadtxt(fname)#, delimiter=',')

n_components = 2
tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0, n_iter=1000, perplexity=20, learning_rate=100, metric='euclidean')

Y = tsne.fit_transform(data)
plt.scatter(Y[:,0], Y[:,1])

plt.show()
