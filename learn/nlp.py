__author__ = 'Aaron J. Masino'


'''
Created on Feb 3, 2014

@author: masinoa
'''

import nltk.data
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
import re

#load a sentence tokenizer, faster for many tokenizations
sent_tokenizer = nltk.data.load('file:./learn/english.pickle')

def sentences(text):
    '''Uses /tokenizers/punkt/english tokenizer to create list of sentences from text'''
    return sent_tokenizer.tokenize(text)

def words(text, splitContractions=False, contractChars = ["'"]):
    '''uses a regexpTokenizer to tokenize text to words. If splitContractions is true,
    the regex pattern is [\w]+ so that contractions are split, e.g. "I can't" -> ['I','can','t'],
    otherwise the regex pattern is [\w']+ so that contractions are not split, i.e. "I can't" -> ['I', "can't"]
    Additional contract characters, e.g. a hyphen, can be added by over riding the contractChars arg'''
    if splitContractions:
        pat = "[\w]+"
    else:
        pat = "[\w{0}]+".format(reduce(lambda x,y: x+y, contractChars, ""))
    return regexp_tokenize(text, pat, discard_empty=True)

def replace_patterns(text, patterns, replaceWith="", flags = None):
    '''replaces each occurrence of each pattern with the replaceWith value'''
    rt = text
    for pat in patterns:
        if not flags:
            rt =  re.sub(pat, replaceWith, rt)
        else:
            rt = re.sub(pat, replaceWith, rt, flags = flags)
    return rt

def replace_whole_words(text, words=[], replaceWith="", flags = re.IGNORECASE):
    rt = text
    for w in words:
        rt = replace_patterns(rt, [r'(?<=\s){0}(?=[\s,!\.;])'.format(w), r'^{0}(?=[\s,!\.;])'.format(w)], replaceWith, flags=flags)
    return rt

def replace_digits(text, replaceWith='number'):
    '''attempts to remove numbers from text using given regex patterns and replace them with the replacementWord'''
    return replace_patterns(text, [r'[-+]?\d+\.\d*',r'[-+]?\d*\.\d+',r'[-+]?\d+'], replaceWith)

def standard_numerals():
    return ["one","two","three","four", "five","six","seven","eight","nine", "eleven","twelve"]

def replace_numerals(text, replaceWith='number', numerals=None, flags = re.IGNORECASE):
    if not numerals:
        return replace_whole_words(text, standard_numerals(), replaceWith, flags)
    else:
        return replace_whole_words(text, numerals, replaceWith, flags)

def standard_units():
    return [r"year", r"years",
            r"month", r"months",
            r"day", r"days",
            r"hour", r"hours",
            r"minute", r"minutes",
            r"second", r"seconds",
            r"cm", r"mm"]

def replace_units(text, replaceWith='unit', units = None, flags = re.IGNORECASE):
    if not units:
        return replace_whole_words(text, standard_units(), replaceWith, flags)
    else:
        return replace_whole_words(text, units, replaceWith, flags)

def porter_stem(wordList):
    porter = nltk.PorterStemmer()
    return [porter.stem(w) for w in wordList]

def filter_stop_words(wordlist):
    sw = stopwords.words('english')
    return filter(lambda w: w not in sw, wordlist)

def text_preprocessor(text):
    ct = replace_digits(text)
    ct = replace_numerals(ct)
    ct = replace_units(ct)
    _words = [word.lower() for word in words(ct)]
    _words = filter(lambda x: x not in stopwords.words('english') and len(x)>2, _words)
    _words = porter_stem(_words)
    return reduce(lambda x,y: '{0} {1}'.format(x,y), _words, "")