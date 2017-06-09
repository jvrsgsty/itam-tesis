#!/bin/python
# coding=utf-8
#
# Counts number of different tags in a collection of files for each part of 
# speech
#
# Author: Javier Sagastuy
# Stand: 08.06.2016
#

import optparse, os, pprint, operator

def countTagsInFile(fname):
  """Counts the number of ocurrences of a a category value in a tagged file. 
  Stores the result in a global dictionary. 

  Args:
    fname(string): full path to the file to process.
  """
  with open(fname, 'r', encoding='utf-8') as f:
    for line in f:
      words = line.split(' ')
      for w in words:
        tag = w.split('_')[1].rstrip()
        cat = tag[0].upper()
        if tag not in dictionaries[cat]:
          dictionaries[cat][tag] = 1
        else:
          dictionaries[cat][tag] += 1

def normalize(w):
  """Normalizes a list of relative weights so it adds up to 1.0 .

  Args:
    w(list): original relative weights. 

  Returns:
      w(list): normalized weights. Adds up to 1.0 .
  """
  s = sum(w)
  for i in range(len(w)):
    w[i] /= s
  return w

def segmentDict(dict, weights):
  """Segments a given category frequency dictionary into a lower number of
  categories, as close as it can to the relative frequencies indicated in 
  weights.

  Args:
    dict(dictionary): original category items are the keys, absolute frequencies
      are teh values. 
    weights(list): list of floats with the relative weights of the target output
      categories.

  Returns:
      [segments, actual_weights]: 
        segments(dictionary): original category items are the keys and the 
          mapped category are the values. 
        actual_weights(list): the actual weights for each of the resulting 
          categories. Should match weights as closely as it can, given the 
          category frequency distribution. 
  """
  # Normalize weights
  weights = normalize(weights)

  segments = {}
  actual_weights = []
  total_instances = 0
  percent_instances = 0
  i = 0
  cat = None

  for k,v in dict.items():
    total_instances += v
    if cat == None:
      cat = k[0].upper()

  sorted_d = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)
  for k,v in sorted_d:
    percent_instances += v/total_instances
    segments[k] = cat + str(i)
    if percent_instances >= weights[i]:
      actual_weights += [percent_instances]
      percent_instances = 0
      i += 1
  actual_weights += [percent_instances]
  return [segments, actual_weights]


if __name__ == '__main__':
  pp = pprint.PrettyPrinter(indent=2)

  categories = {}
  categories['A'] = 'adjetivo'
  categories['R'] = 'adverbio'
  categories['D'] = 'determinante' # articulo
  categories['N'] = 'nombre' # sustantivo
  categories['V'] = 'verbo'
  categories['P'] = 'pronombre'
  categories['C'] = 'conjunción'
  categories['I'] = 'interjección'
  categories['S'] = 'preposición'
  categories['F'] = 'puntuación'
  categories['Z'] = 'numeral'
  categories['W'] = 'fecha'
  categories['3'] = '?????'

  dictionaries = {}

  dictionaries['A'] = {}
  dictionaries['R'] = {}
  dictionaries['D'] = {}
  dictionaries['N'] = {}
  dictionaries['V'] = {}
  dictionaries['P'] = {}
  dictionaries['C'] = {}
  dictionaries['I'] = {}
  dictionaries['S'] = {}
  dictionaries['F'] = {}
  dictionaries['Z'] = {}
  dictionaries['W'] = {}
  dictionaries['3'] = {}

  path = '../data/tagged/'
  for fname in os.listdir(path):
    ftype = fname.split('.')[1]
    if ftype == 'tagged':
      countTagsInFile(path + fname)

  pp.pprint(dictionaries)

  total_tokens = 0
  total_categorias = 0
  number_of_tokens = {}
  for k,v in dictionaries.items():
    total_categorias += len(v)
    num_tokens = 0
    for key,value in v.items():
      num_tokens += value
    total_tokens += num_tokens
    number_of_tokens[k] = num_tokens

  print('-'*80)
  print('Categoría | # Instancias | % categorías | # tokens | % tokens | %c * %t')
  print('-'*80)
  for k,v in sorted(number_of_tokens.items()):
    percent_tok = 100*(v/total_tokens)
    percent_cat = 100*len(dictionaries[k])/total_categorias
    string = ' '*5 + k + ' '*4
    string += '|{:8d}'.format(len(dictionaries[k])) + ' '*6
    string += '|{:8.2f}'.format(percent_cat) + ' '*6
    string += '|{:7d}'.format(v) + ' '*3
    string += '| {:6.2f}'.format(percent_tok) + ' '*3
    string += '| {:6.2f}'.format((percent_tok * percent_cat))
    print(string)

  print('-'*80)
  print('Testing the segmentation function...')
  print('-'*80)

  weights = [1,1,1,1]
  [s, aw] = segmentDict(dictionaries['V'], weights)
  pp.pprint(sorted(s.items(), key=operator.itemgetter(1)))
  print('Target Weights: ' + str(normalize(weights)))
  print('Actual Weights: ' + str(aw))
