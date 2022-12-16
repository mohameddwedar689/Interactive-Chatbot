# Used lib 
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tensorflow as tf
import tflearn
import random
import json
import pickle
from time import sleep


# Open Intents file and load data from it
with open('intents.json') as file:
    data = json.load(file)


# Create many list to adding data it For shaping purposes
# this list for words drived from tokenization process
words = []
# this list for all tages that that bot can detect it
lables = []
# this is a document list for words without tokenization
docs_x = []
# this is a list that contain the relationship between words and tages
docs_y = []

# Looping on intents to append data to lists
for intent in data['intents']:
    for patterns in intent['patterns']:
        w = nltk.word_tokenize(patterns)
        words.extend(w)
        docs_x.append(w)
        docs_y.append(intent['tag'])

    if intent['tag'] not in lables:
        lables.append(intent['tag'])





