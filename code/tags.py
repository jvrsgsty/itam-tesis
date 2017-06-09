#!/bin/python
# coding=utf-8
#
# Parses spanish tagged files by the freeling and Stanford
# POS taggers and interprets the tags to print only the
# part of speech of each word in the input file
#
# Author: Javier Sagastuy
# Stand: 28.05.2016
#

import optparse, ntpath

def parse_freeling(fname):
  with open(fname, encoding = "ISO-8859-1") as f:
    for line in f:
      if line != '\n':
        tag = line.split(' ')[2]
        cat = tag[0].upper()
        print(categories[cat])

def parse_stanford(fname):
  with open(fname, 'r', encoding='utf-8') as f:
    for line in f:
      words = line.split(' ')
      for w in words:
        tag = w.split('_')[1]
        cat = tag[0].upper()
        print(categories[cat])

if __name__ == '__main__':
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

  parser = optparse.OptionParser()
  parser.add_option("-f", "--file", dest="filename",
                  help="File to analyze", metavar="FILE")
  (options, args) = parser.parse_args()

  fname = options.filename
  ftype = ntpath.basename(fname).split('.')[1]
  if ftype == 'mrf':
    parse_freeling(fname)
  elif ftype == 'tagged':
    parse_stanford(fname)
