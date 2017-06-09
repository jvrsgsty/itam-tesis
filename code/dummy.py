#!/bin/python
# coding=utf-8
#
# Splits a mixed database's categorical columns into dummy columns and
# exports the resulting db
#
# Author: Javier Sagastuy
# Stand: 07.05.2017
#

import os
import numpy as np
import pandas as pd

def generateDummy(fname):
  data = pd.read_table(fname, header=None)
  categorical = []
  numerical = []
  for k in range(data.shape[1]):
    if data[k].dtype == 'O':
      categorical += [k]
    if 'float' in str(data[k].dtype) or 'int' in str(data[k].dtype):
      numerical += [k]

  categorical.sort(reverse=True)
  numerical.sort(reverse=True)

  num_dummy = 0
  for c in categorical:
    num_dummy += len(data[c].unique())

  dummy = np.zeros((data.shape[0], (data.shape[1]-len(categorical))+num_dummy))
  dummy = stabilize(dummy)
  col = 0
  while len(categorical)>0 and len(numerical)>0:
    if categorical[-1] < numerical[-1]:
      c = categorical.pop()
      for inst in data[c].unique():
        for k in range(len(data[c])):
          if data[c][k] == inst:
            dummy[k, col] += 1
        col += 1
    else:
      c = numerical.pop()
      dummy[:,col] = np.array(data[c])
      col += 1

  if(len(categorical) > 0):
    categorical.reverse()
    for c in categorical:
      for inst in data[c].unique():
        for k in range(len(data[c])):
          if data[c][k] == inst:
            dummy[k, col] += 1
        col += 1
  else:
    numerical.reverse()
    for c in numerical:
      dummy[:,col] = np.array(data[c])
      col += 1
  return dummy

def stabilize(data, order=-6):
  data += np.random.rand(data.shape[0], data.shape[1]) * 10**order
  return data

def exportNumpy(nparray, base_path, fname):
  # Numpy writes to textfile in byte mode
  with open(base_path + fname, 'wb') as f:
    np.savetxt(f, nparray, delimiter="\t")

if __name__ == '__main__':
  base_path = '../Batches/BatchH1_JSB/data/DBs/'

  for f in os.listdir(base_path):
    fname = f.split('.')[0]
    ftype = f.split('.')[1]
    if ftype == 'dat':
      dummy = generateDummy(base_path + f)
      exportNumpy(dummy, base_path + '../dummy_DBs/', fname + '_dummy.' + ftype)

