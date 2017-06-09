#!/bin/python
# coding=utf-8
#
# Creates test subsets of the large original DBs for testing
#
# Author: Javier Sagastuy
# Stand: 03.04.2017
#

import os

NUM_LINES = 1000

def splitDB(base_path, f_in, size):
  fname = f_in.split('.')[0]
  ftype = f_in.split('.')[1]
  f_out = open(base_path + str(NUM_LINES) +'/' + fname + '_' + str(size) + '.' + ftype,
               'w', encoding='utf-8')
  with open(base_path + f_in, 'r', encoding='utf-8') as f:
    i = 0
    for line in f:
      f_out.write(line)
      i += 1
      if i >= size:
        break
  f_out.close()

if __name__ == '__main__':
  base_path = '../data/DBs/12t_30p_NA_diferentes/'
  for f in os.listdir(base_path):
    if os.path.isfile(base_path + f):
      fname = f.split('.')[0]
      ftype = f.split('.')[1]
      if ftype == 'dat':
        wc_out = os.popen('wc -l ' + base_path + f).read()
        num_lines = int(wc_out.split()[0])
        print(num_lines)
        if(num_lines >= NUM_LINES):
        # for size in range(100,1001,100):
            splitDB(base_path, f, NUM_LINES)

