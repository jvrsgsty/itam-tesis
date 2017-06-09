#!/bin/python
# coding=utf-8
#
# Meassures pair-wise similarity for encoded DBs in a directory
#
# Author: Javier Sagastuy
# Stand: 21.05.2017
#

import os, optparse, itertools, random, progressbar, csv, pickle, math
from subprocess import DEVNULL, STDOUT, check_call
from multiprocessing import Process, Manager
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from scipy import misc, spatial
from rpy2.robjects import FloatVector, r
from rpy2.robjects.packages import importr
from rpy2.robjects.numpy2ri import numpy2ri
kohonen = importr('kohonen')
e1071 = importr('e1071')
sns.set_context('talk')

###################
# Imports / Exports
def exportResults(fname, headers, sim_matrix):
  with open(fname, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
  # Numpy writes to textfile in byte mode
  N = len(headers)
  with open(fname, 'ab') as f:
    np.savetxt(f, sim_matrix, delimiter=",")

def importObject(fname):
  return pickle.load(open( fname, 'rb'))

def exportObject(res, fname):
  pickle.dump(res, open(fname, 'wb'))

###################
# Clustering
def calculateMembership_CBE(data, k, fname):
  CBE_PATH = "CBE_JSB/"
  RESULTS_PATH = os.path.dirname(fname) + '/cbe/'
  text = os.path.basename(fname).split('.')[0]
  f_in = fname
  # f_out must have txt extension for CBE
  f_out = RESULTS_PATH + text + "_cbe_results.txt"
  #with open(f_in, 'wb') as f:
  #  np.savetxt(f, data, delimiter="\t")
  args = str(k) + " " + f_in + " " + f_out
  classpath = "-cp " + CBE_PATH
  java_call = "java " + classpath + " CBE " + args
  exit_code = check_call(java_call, shell=True, stdout=DEVNULL, stderr=STDOUT)
  fname = f_out.replace(".txt", "_cluster_labels.txt")
  labels_one_index = np.loadtxt(fname, delimiter=',', dtype='int8')
  zero_index = np.vectorize(lambda x: x-1)
  return zero_index(labels_one_index)

def fuzzyCMeans(data, k):
  data_train_matrix = numpy2ri(data)
  results = e1071.cmeans(data_train_matrix, k)
  centers = np.array(results.rx2('centers'))
  membership = np.array(results.rx2('membership'))
  withinerror = np.array(results.rx2('withinerror'))
  return withinerror

def calculateCenters(data, k):
  # TODO: use k to define the topology. Linear topology is not always best
  data_train_matrix = numpy2ri(data)
  # Fix the seed, since for the same text, the centers will be different on
  # two independent runs, yielding different membership assignments
  r('set.seed')(12345)
  grid = kohonen.somgrid(xdim=k, ydim=1, topo="rectangular")

  kwargs = {'grid': grid,
            'rlen': 500,
            'radius': FloatVector([4.0, 0.3262874458]),
            'alpha': FloatVector([0.999, 0.0065639126]),
            'mode': 'batch',
            'dist.fcts': 'euclidean'}

  results = kohonen.som(data_train_matrix, **kwargs)

  centers = np.array(results.rx2('codes')[0])
  return centers

def calculateMembership(data, centers):
  distances = spatial.distance.cdist(data, centers)
  return np.argmin(distances, axis=1)

def clustering(fname, k, method):
  if method == 'soms':
    data = np.loadtxt(fname)#, delimiter=',')
    centers = calculateCenters(data, k)
    return calculateMembership(data, centers)
  elif method == 'cbe':
    if os.path.isfile(fname.replace('.dat', '_cbe_membership.list')):
      membership = importObject(fname.replace('.dat', '_cbe_membership.list'))
    else:
      data = np.loadtxt(fname)#, delimiter=',')
      membership = calculateMembership_CBE(data, k, fname)
      exportObject(membership, fname.replace('.dat', '_cbe_membership.list'))
    return membership
  else:
    return clustering('soms', fname, k)

def clustering_worker(fname, k, method, parallel_out, idx):
  parallel_out[idx] = clustering(fname, k, method)

###################
# Cluster Matching
def findBestPermutation(k, m1, m2):
  best_idx_p = 0
  best_idx_n = 0
  min_non_zero = len(m1)
  min_norm = np.linalg.norm(np.ones(len(m1))*len(m1))

  # Preliminary variables for frequencies approach
  y = np.bincount(m1)
  ii = np.nonzero(y)[0]
  freq1 = y[ii]

  y = np.bincount(m2)
  ii = np.nonzero(y)[0]
  freq2 = y[ii]

  # randomly pick at most 1000 permutations of the indices
  if k >= 7:
    # 7! = 5040
    n = 1000
  else:
    n = math.factorial(k)
  # Nevermind... Always sample 100 times.
  n = 1000
  for i in range(n):
    perm = np.random.permutation(k)
    # Coincidence probability approach
    vf = np.vectorize(lambda x: perm[x])
    mp = vf(m2)
    non_zero = np.count_nonzero(m1 - mp)
    if non_zero < min_non_zero:
      min_non_zero = non_zero
      best_perm_p = perm
    # Frequencies approach
    freqp = freq2[perm]
    norm = np.linalg.norm(freq1 - freqp)
    if norm < min_norm:
      min_norm = norm
      best_perm_n = perm
  coincidences = len(m1) - min_non_zero
  probability = {'coincidences': coincidences, 'perm': best_perm_p}
  frequencies = {'norm': min_norm, 'perm': best_perm_n}
  best = {'probability': probability, 'frequencies': frequencies}
  return best

def coincidenceProbability(m, k, n):
  s = 0
  for i in range(m, n+1):
    s += misc.comb(n, i, exact = True)*(k-1)**(n-i)
  return 1 - (s / k**n)

def similarity(m1, m2, k):
  n = len(m1)
  best = findBestPermutation(k, m1, m2)
  # Coincidence probaility
  m = best['probability']['coincidences']
  s = coincidenceProbability(m, k, n)
  # Frequencies
  norm = best['frequencies']['norm']
  return {'coincidences': m, 'probability': s, 'frequencies': norm}

####################
# Flows
def sim_flow(base_path, k=3, method='soms', graph=False):
  print('Using ' + method + ' clustering method with ' + str(k) + ' clusters')
  files = []
  for f in os.listdir(base_path):
    if not os.path.isdir(os.path.join(base_path, f)):
      fname = f.split('.')[0]
      ftype = f.split('.')[1]
      if ftype == 'dat':
        files += [f]
  #################
  # Clustering
  N = len(files)
  print("Clustering:")
  bar = progressbar.ProgressBar(maxval=N, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
  bar.start()
  manager = Manager()
  cluster_labels = manager.dict()
  threads = []
  j = 0
  for i in range(N):
    args = (base_path + files[i], k, method, cluster_labels, i)
    t = Process(target=clustering_worker, args=args)
    threads.append(t)
    t.start()
    if len(threads) >= 4:
      for t in threads:
        t.join()
        j += 1
        bar.update(j)
      threads = []
  # If any threads are still running, wait for them to finish
  if len(threads) > 0:
    for t in threads:
      t.join()
      j += 1
      bar.update(j)
    threads = []
  bar.finish()

  #################
  # Find best match and calculate probabilities
  print("Cluster matching:")
  bar = progressbar.ProgressBar(maxval=(N**2-N)/2+N, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
  bar.start()
  sim_matrix = np.zeros((N,N))
  sim_matrix2 = np.zeros((N,N))
  sim_matrix3 = np.zeros((N,N))
  jj = 0
  for i in range(N):
    for j in range(i, N):
      s = similarity(cluster_labels[i], cluster_labels[j], k)
      sim_matrix[i,j] = s['probability']
      sim_matrix2[i,j] = s['frequencies']
      sim_matrix3[i,j] = s['coincidences']
      jj += 1
      bar.update(jj)
  bar.finish()

  sim_matrix = sim_matrix + np.transpose(sim_matrix) - np.eye(N)
  sim_matrix2 = sim_matrix2 + np.transpose(sim_matrix2)
  tuples = len(cluster_labels[0])
  sim_matrix3 = sim_matrix3 + np.transpose(sim_matrix3) - tuples*np.eye(N)

  exportResults(base_path + 'sim_probability_' + method + '.csv',
                headers=files, sim_matrix=sim_matrix)
  exportResults(base_path + 'sim_frequencies_' + method + '.csv',
                headers=files, sim_matrix=sim_matrix2)
  exportResults(base_path + 'sim_coincidences_' + method + '.csv',
                headers=files, sim_matrix=sim_matrix3)

  if graph:
    graphHeatmaps(sim_matrix, sim_matrix2, sim_matrix3, method)
    graphSimilarity(sim_matrix, 0, files)

###################
# Graphs

def graphCoincidenceProbability():
  k = 5
  m = 100
  x = np.arange(0, m, 1)
  f = lambda x: coincidenceProbability(x, k, m)
  y = np.array([f(xi) for xi in x])
  sns.set_context('talk', rc={"lines.linewidth": 3.0})
  ax = plt.plot(x, y, '-')
  plt.ylim([min(y)-0.1*max(y), max(y)*1.1])
  #plt.title('Similitud por probabilidad de \n número de coincidencias', fontsize=24)
  plt.xlabel(r'$m_{ij}$', fontsize=30)
  plt.ylabel(r'$\zeta_{ij}$', fontsize=30)
  plt.tick_params(axis='both', which='major', labelsize=20)
  plt.tight_layout()
  plt.savefig('coincidence_probability_m'+ str(m) +'.eps', format='eps', dpi=1000)
  plt.show()

def graphSimilarity(data, idx, labels):
  y = data[idx,:].tolist()
  del y[idx]
  x = range(len(y))
  plt.plot(x, y, 'bo')
  plt.plot((sum(y)/len(y))*np.ones(len(y)), 'r-')
  plt.title('Similitud')
  plt.ylabel('Similitud')
  plt.ylim([min(y)-0.1*max(y), max(y)*1.1])
  plt.xlim([min(x)-0.1*max(x), max(x)*1.1])
  plt.show()

def graphElbow(fname):
  graphElbowFuzzy(fname)
  graphElbowSoms(fname)
  plt.show()

def graphElbowFuzzy(fname):
  data = np.loadtxt(fname)
  x = range(2,41)
  y = np.array([fuzzyCMeans(data, k) for k in x])
  f = plt.figure(1)
  plt.plot(x, y, 'o', c=sns.color_palette()[0])
  plt.plot(x, y, '-', c=sns.color_palette()[0])
  plt.ylabel('Within-error')
  plt.title('Elbow Graph: Fuzzy C-Means')
  plt.savefig('elbow_fuzzy.eps', format='eps', dpi=1000)
  f.show()
  graphDelta(x, y, 2, 'Elbow Graph: Fuzzy C-Means', 'fuzzy')

def graphElbowSoms(fname):
  data = np.loadtxt(fname)
  x = range(2,41)
  all_centers = [calculateCenters(data, k) for k in x]
  # http://stackoverflow.com/questions/6645895/calculating-the-percentage-of-variance-measure-for-k-means
  D_k = [spatial.distance.cdist(data, centers, 'euclidean') for centers in all_centers]
  cIdx = [np.argmin(D,axis=1) for D in D_k]
  dist = [np.min(D,axis=1) for D in D_k]
  avgWithinSS = [sum(d**2)/data.shape[0] for d in dist]
  y = avgWithinSS
  f = plt.figure(3)
  plt.plot(x, y, 'o', c=sns.color_palette()[0])
  plt.plot(x, y, '-', c=sns.color_palette()[0])
  #plt.ylabel(r'$WCSS_k$', fontsize=30)
  plt.xlabel(r'$k$: Número de clusters', fontsize=30)
  #plt.title('Elbow Graph: Clustering with SOMs')
  plt.savefig('elbow_soms.eps', format='eps', dpi=1000)
  f.show()
  graphDelta(x, y, 4, 'Elbow Graph: Clustering with SOMs', 'soms')

def graphDelta(x, y, fig_idx, title, method):
  deltas = np.array(y[0:-2]) - np.array(y[1:-1])
  f = plt.figure(fig_idx)
  x = x[1:-1]
  plt.plot(x, deltas, 'o', c=sns.color_palette()[0])
  plt.plot(x, deltas, '-', c=sns.color_palette()[0])
  #plt.ylabel(r'$WCSS_{k-1}-WCSS_k$', fontsize=30)
  plt.xlabel(r'$k$: Número de clusters', fontsize=30)
  #plt.title(title)
  plt.savefig('elbow_delta_' + method + '.eps', format='eps', dpi=1000)
  f.show()

def graphHeatmaps(sim_matrix, sim_matrix2, sim_matrix3, method):
  cmap_r = ListedColormap(sns.color_palette('RdBu_r', 32).as_hex())
  cmap = ListedColormap(sns.color_palette('RdBu', 32).as_hex())
  f = plt.figure(1)
  sns.heatmap(sim_matrix, cmap=cmap, xticklabels=False, yticklabels=False)
  #plt.title('Similitud medida usando probabilidad de coincidencias')
  plt.savefig('heatmap_probability_' + method + '.eps', format='eps', dpi=1000)
  f.show()
  f = plt.figure(2)
  sns.heatmap(sim_matrix2, cmap=cmap_r, xticklabels=False, yticklabels=False)
  #plt.title('Similitud medida usando distancia euclidiana')
  plt.savefig('heatmap_frequencies_' + method + '.eps', format='eps', dpi=1000)
  f.show()
  f = plt.figure(3)
  vmax = np.setdiff1d(sim_matrix3, np.array(sim_matrix3.max())).max()
  sns.heatmap(sim_matrix3, vmax=vmax, cmap=cmap, xticklabels=False, yticklabels=False)
  #plt.title('Similitud medida usando número de coincidencias')
  plt.savefig('heatmap_coincidences_' + method + '.eps', format='eps', dpi=1000)
  f.show()
  plt.show()

def graphIdealHeatmap(directory):
  files = []
  for f in os.listdir(base_path):
    if not os.path.isdir(os.path.join(base_path, f)):
      fname = f.split('.')[0]
      ftype = f.split('.')[1]
      if ftype == 'dat':
        files += [f]
  n = len(files)
  sim_matrix = np.zeros((n,n))
  for i in range(n):
    for j in range(i, n):
      if files[i][0:2] == files[j][0:2]:
        sim_matrix[i,j] = 1
  sim_matrix = sim_matrix + sim_matrix.transpose() - np.eye(n)
  cmap = ListedColormap(sns.color_palette('RdBu', 32).as_hex())
  sns.heatmap(sim_matrix, cmap=cmap, xticklabels=2, yticklabels=2)
  plt.savefig('ideal_heatmap.eps', format='eps', dpi=1000)
  plt.show()


####################
# Main
if __name__ == '__main__':
  parser = optparse.OptionParser()
  parser.add_option("-d", "--directory", dest="base_path",
                  help="Directory to analyze", metavar="FILE")
  parser.add_option('-m', '--method', dest='method', default='soms',
                    action='store', help='Clustering method to use')
  parser.add_option('-k', '--k', dest='k', type="int", default=3,
                    action='store', help='Number of clusters')
  parser.add_option('-f', '--flow', dest='flow', default='sim',
                    action='store', help='sim | elbow | prob')
  parser.add_option('-g', '--graph', dest='graph', default=False,
                    action='store_true',
                    help='Presence indicates to complement sim flow results with a graph')

  (options, args) = parser.parse_args()
  # Other defaults
  base_path = options.base_path
  if base_path == None:
    base_path = '../Batches/BatchT1_JSB/data/cbe_test/'

  if options.flow == 'sim':
    print('Clustering and calculating pair-wise similarity for all files...')
    sim_flow(base_path, k=options.k, method=options.method, graph=options.graph)
  elif options.flow == 'elbow':
    print('Generating elbow graphs...')
    graphElbow(base_path)
  elif options.flow == 'iheat':
    print('Generating ideal heatmap...')
    graphIdealHeatmap(base_path)
  elif options.flow == 'prob':
    print('Generating coincidence probability graph...')
    graphCoincidenceProbability()
