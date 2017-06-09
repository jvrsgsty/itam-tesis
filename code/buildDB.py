#!/bin/python
# coding=utf-8
#
# Creates the categorical DB from a tagged document into the format accepted
# by CENG
#
# Author: Javier Sagastuy
# Stand: 26.03.2017
#

from collections import Counter
import optparse, os, pprint, operator
import countTags as ct

NUM_NUMERICAL = 3   # Number of numerical variables in the zentence structure
NUM_TOKENS = 12     # Max number of tokens to analyze in each zentence
PERCENT_NA = 0.3    # Max percentage of NAs admissible in a zentence

def isFull(zentence, num_tokens):
  """Determines whether or not a given zentence can be considered full

  Args:
    zentence(dict): a zentence dictionary where values not yet filled should
     be equal to 'NA'
    num_tokens(int): How many tokens have been analyzed to fill the current
      zentence up
  """
  c = Counter(zentence.values())
  is_full = c['NA']/(len(zentence) - NUM_NUMERICAL) <= PERCENT_NA
  is_full = is_full or num_tokens > NUM_TOKENS
  return is_full

def isWord(tag):
  return tag[0].upper() not in ['F', 'Z', 'W', '3']

def isPunctuation(tag):
  return tag[0].upper() == 'F'

def uniquifyNAs(zentence):
  for k,v in zentence.items():
    if v == 'NA':
      zentence[k] += '_' + k
  return zentence

def buildDB(fname, mappings):
  """Builds a structured database from a set of given filenames.

  Args:
    paths(list): items are full paths to the files to parse into a structured
      database.
  """
  database = []
  all_words = []
  # Hoping the file fits in memory...
  with open(fname, 'r', encoding='utf-8') as f:
    for line in f:
      words = line.split(' ')
      all_words += words

  i = 0
  # Iterate over all the words in a text
  while i < len(all_words):
    zentence = {'A': 'NA', 'C': 'NA', 'D': 'NA', 'F': 'NA', 'N_1': 'NA',
                'N_2': 'NA', 'P': 'NA', 'RS': 'NA', 'V': 'NA', 'IWZ': 'NA',
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
      # Overwrite tag with its mapping if there is one
      if cat in mappings.keys():
        tag = mappings[cat][tag]

      j = 0
      l = len(keys) - NUM_NUMERICAL
      is_filled = False
      # For a given word, iterate through the whole zentence and try to make
      # it fit
      # j < l is always true since numeric columns are never updated
      while not is_filled and j < l:
        word_cats = keys[j].split("_")[0]
        if cat in word_cats and zentence[keys[j]] == 'NA':
          # TODO: Should we append column index to make instances in N_1
          #       different from thos in N_2? CENG would treat them as the
          #       same instance
          zentence[keys[j]] = tag
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
      # Append category name to NA values
      zentence = uniquifyNAs(zentence)
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
  parser = optparse.OptionParser()
  parser.add_option("-f", "--file", dest="filename",
                  help="File to analyze", metavar="FILE")
  (options, args) = parser.parse_args()

  pp = pprint.PrettyPrinter(indent=2)
  base_path = '../data/tagged/'

  dictionaries = {}
  categories = ['A','R','D','N','V','P','C','I','S','F','Z','W','3']
  for c in categories:
    dictionaries[c] = {}

  countTags(dictionaries)

  # Segment categories according to predefined weights
  mappings = {}
  # Decided not to segment, to guarantee same granularity
  # k = 'V'
  # weights = [1,1,1,1,1,1]
  # [s, actual_weights] = ct.segmentDict(dictionaries[k], weights)
  # mappings[k] = s
  # k = 'D'
  # weights = [1,1,1,1]
  # [s, actual_weights] = ct.segmentDict(dictionaries[k], weights)
  # mappings[k] = s
  # k = 'F'
  # weights = [1,1,1,1]
  # [s, actual_weights] = ct.segmentDict(dictionaries[k], weights)
  # mappings[k] = s
  # k = 'P'
  # weights = [1,1,1]
  # [s, actual_weights] = ct.segmentDict(dictionaries[k], weights)
  # mappings[k] = s

  # Only export specified file
  if options.filename:
    fname = options.filename
    file_path = base_path + fname + '.tagged'
    if os.path.exists(file_path):
      db = buildDB(file_path, mappings)
      print('Exporting ' + fname + '...')
      exportDB(db, base_path + '../DBs/12t_30p/' + fname + '.dat')
      print('DONE')
    else:
      print('File: ' + file_path + 'does not exist')
  # If no file is specified, export all
  else:
    for f in os.listdir(base_path):
      fname = f.split('.')[0]
      ftype = f.split('.')[1]
      if ftype == 'tagged':
        file_path = base_path + fname + '.tagged'
        db = buildDB(file_path, mappings)
        print('Exporting ' + fname + '...')
        exportDB(db, base_path + '../DBs/12t_30p_NA_diferentes/' + fname + '.dat')
        print('DONE')
