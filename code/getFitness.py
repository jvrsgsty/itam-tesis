#!/bin/python
import os, pprint, operator
fitness = {}
path = '../data/encoded_DBs/'
for f in os.listdir(path):

  fname = f.split('.')[0]
  if len(f.split('.')) > 1:
    ftype = f.split('.')[1]
  else:
    ftype = 'none'
  if ftype == 'out' and fname != "times":
    rens = int(fname.split('_')[2])
    cols = int(fname.split('_')[0])
    if not cols in fitness.keys():
      fitness[cols] = {}
    with open(path + f, 'r', encoding='utf-8') as f:
      line = f.readline()
      fitness[cols][rens] = line.split(':')[1]

#pp = pprint.PrettyPrinter(indent=2)
#pp.pprint(fitness)

s = ''
c = ''
first = True
sorted_f = sorted(fitness.items(), key=operator.itemgetter(0), reverse=False)
print(sorted_f)
for k,d in sorted_f:
  s += str(k) + ','
  sorted_d = sorted(d.items(), key=operator.itemgetter(0), reverse=False)
  for kk, v in sorted_d:
    if first:
      c += str(kk) + ','
    v = v[:-2]
    s += v + ','
  first = False
  s+= '\n'
print(c)
print(s)
