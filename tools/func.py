from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import time
import re
import string

import tensorflow as tf 
import numpy as np
import pandas as pd


import nltk
# uncomment line below for the first time to download the nltk packages
#nltk.download()
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk import pos_tag 


_porter = PorterStemmer()
_lemmatizer = WordNetLemmatizer()



#Stopword
_stop_words = set(stopwords.words('english')) 

#Flatten Function
flatten = lambda l: [item for sublist in l for item in sublist]

class main_function:
    def __init__(self):
        pass
    #Remove punc Function
    def remove_punc(self,txt):
        return "".join([t for t in txt if t not in string.punctuation])

    #Split sentense to list and remove non text
    def tokenize(self,df,regex=r"[a-zA-Z]+"):
        #Regex default match only word
        return RegexpTokenizer(regex).tokenize(text=df.lower())

    #Lemmatizing to reduce inflectional forms 
    def word_lemmatizer(self,txt,tag='n'):
        # default change to nounce
        return  [ _lemmatizer.lemmatize(word,tag)  for word in txt  ]


    #filter Stem Words
    def word_stem(self,txt):
        return [_porter.stem(word) for word in txt]

    #filter Stopword
    def stop_word(self,txt):
        filtered_sentence = [] 
        for w in txt: 
            if w not in _stop_words: 
                filtered_sentence.append(w) 
        return filtered_sentence 

    #Convert word to grammar tag (Return tag Only)
    def pos_tagging(self,txt): 
        return [x[1] for x in pos_tag(txt)]