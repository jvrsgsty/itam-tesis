#!/bin/python
# coding=utf-8
#
# Creates the categorical DB from a tagged document into the format accepted
# by CENG.
# Format is hardcoded for the parametrization runs, using only verbs
#
# Author: Javier Sagastuy
# Stand: 11.06.2016
#

from collections import Counter
import optparse, os, pprint, operator
import countTags as ct

def isFull(zentence, num_tokens):
  c = Counter(zentence.values())
  return num_tokens > 400 or c['NA']/len(zentence) < 0.002

def isWord(tag):
  return tag[0].upper() not in ['F', 'Z', 'W', '3']

def isPunctuation(tag):
  return tag[0].upper() == 'F'

def buildDB(paths, mappings):
  """Builds a structured database from a set of given filenames.

  Args:
    paths(list): items are full paths to the files to parse into a structured 
      database. 
  """
  database = []
  all_words = []
  # Hoping the file fits in memory... 
  for fname in paths:
    with open(fname, 'r', encoding='utf-8') as f:
      for line in f:
        words = line.split(' ')
        all_words += words

  i = 0
  # Iterate over all the words in a text
  while i < len(all_words):
    # zentence = {'A': 'NA', 'R': 'NA', 'D': 'NA', 'N': 'NA', 'V': 'NA', 
    #             'P': 'NA', 'C': 'NA', 'I': 'NA', 'S': 'NA', 'F': 'NA', 
    #             'Z': 'NA', 'W': 'NA', 
    #             '_tokens': 0, '_words': 0, '_punct': 0}
    zentence = {'V_1': 'NA', 'V_2': 'NA', 'V_3': 'NA', 'V_4': 'NA', 
                '_tokens': 0, '_words': 0, '_punct': 0}
    num_tokens = 0
    num_words = 0
    num_punctuation = 0
    keys = sorted(list(zentence.keys()))
    # Fill a zentence up
    while not isFull(zentence, num_tokens) and i < len(all_words):
      w = all_words[i]
      tag = w.split('_')[1].rstrip()
      cat = tag[0].upper()
      # Overwrite tag with its mapping
      tag = mappings[cat][tag]

      j = 0
      l = len(keys)
      is_filled = False
      # j < l is always true since numeric columns are never updated
      while not is_filled and j < l:
        word_cat = keys[j][0]
        if cat == word_cat and zentence[keys[j]] == 'NA':
          # Append the index, to make a different instance in new column
          zentence[keys[j]] = tag + '_' + str(j)
          is_filled = True
        j += 1

      if isWord(tag):
        num_words += 1
      if isPunctuation(tag):
        num_punctuation += 1
      num_tokens += 1
      i += 1
    # Once a zentence is full, add the numeric values
    if isFull(zentence, num_tokens):
      zentence['_tokens'] = num_tokens
      zentence['_words'] = num_words
      zentence['_punct'] = num_punctuation
      database += [zentence]
  return database

def prettyPrint(database):
  """Prints a given database on the console in a readable format.

  Args:
    database(list): items are zentences i.e. dictionaries where the key is 
      the name of the column and the value is the category or numerical 
      value for that column. 
      frequency dictionaries where keys are category values and values are 
      absolute frequency counts. 
  """
  for zentence in database:
    s = ''
    for k,v in sorted(zentence.items()):
      s += str(v) + "\t"
    print(s)

def exportDB(database, fname):
  """Writes a parsed database to a tab separated file. 

  Args:
    database(list): items are zentences i.e. dictionaries where the key is 
      the name of the column and the value is the category or numerical 
      value for that column. 
      frequency dictionaries where keys are category values and values are 
      absolute frequency counts. 
    fname(string): full output path of the file to write. 
  """
  f = open(fname, 'w', encoding='utf-8')
  for zentence in database:
    s = ''
    for k,v in sorted(zentence.items()):
      s += str(v) + '\t'
    # We remove the final tab
    s = s[:-1]
    f.write(s + '\n')
  f.close()

def countTags(dictionaries):
  """Counts the number of ocurrences of a a category value in all tagged files. 
  Stores the result in a given dictionary and returns it.

  Args:
    dictionaries(dictionary): keys are category initials. Values are 
      frequency dictionaries where keys are category values and values are 
      absolute frequency counts. 

  Returns:
    dictionaries(dictionary): updated input dictonaries.
  """
  path = '../data/tagged/'
  for fname in os.listdir(path):
    ftype = fname.split('.')[1]
    if ftype == 'tagged':
      dictionaries = countTagsInFile(path + fname, dictionaries)
  return dictionaries

# Overrides the definition in countTags.py
def countTagsInFile(fname, dictionaries):
  """Counts the number of ocurrences of a a category value in a tagged file. 
  Stores the result in a given dictionary and returns it. 

  Args:
    fname(string): full path to the file to process.
    dictionaries(dictionary): keys are category initials. Values are 
      frequency dictionaries where keys are category values and values are 
      absolute frequency counts. 

  Returns:
    dictionaries(dictionary): updated input dictonaries.
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
  return dictionaries

if __name__ == '__main__':
  pp = pprint.PrettyPrinter(indent=2)
  base_path = '../data/tagged/'

  dictionaries = {}
  categories = ['A','R','D','N','V','P','C','I','S','F','Z','W','3']
  for c in categories:
    dictionaries[c] = {}

  countTags(dictionaries)

  # Segment all categories with the same weights (4 instances per category)
  weights = [1,1,1,1]
  mappings = {}
  for k,v in dictionaries.items():
    [s, aw] = ct.segmentDict(v, weights)
    mappings[k] = s

  paths = []
  for f in os.listdir(base_path):
    fname = f.split('.')[0]
    ftype = f.split('.')[1]
    if ftype == 'tagged':
      paths += [base_path + fname + '.tagged']

  db = buildDB(paths, mappings)

  #prettyPrint(db)
  print('Exporting...')
  exportDB(db,base_path + '../DBs/' + fname + '.dat')
  print('DONE')
