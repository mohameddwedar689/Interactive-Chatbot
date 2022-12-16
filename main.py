# Used lib 
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
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


# Stemming data form words list
words = [stemmer.stem(w.lower()) for w in words if w != '?']
# for overfeding purposes
words = sorted(list(set(words)))
lables = sorted(lables)



#print(words)
#print(lables)


# Create training list
training = []
output = []

out_empty = [0 for _ in range(len(lables))]

#print(out_empty)


for x , doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w) for w  in doc]

    #print(wrds)

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)


    #print(bag)

    output_row = out_empty[:]
    output_row[lables.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

#print(training)
#print(output)
 
# convert it to numpy array 
training = numpy.array(training)
output = numpy.array(output)



