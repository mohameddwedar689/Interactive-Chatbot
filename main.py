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

try:
    with open('data.pickle' , "rb") as f:
        words , lables , training , output  = pickle.load(f)
except:
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

    with open('data.pickle' , 'wb') as f:
        pickle.dump((words , lables , training , output) , f)



# Training Model

net = tflearn.input_data(shape=[None , len(training[0])])
net = tflearn.fully_connected(net , 8)
net = tflearn.fully_connected(net , 8)
net = tflearn.fully_connected(net , len(output[0]) , activation="softmax")
net = tflearn.regression(net)

# select nueral network
model = tflearn.DNN(net)
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


# Function that convert user statement to bag of words
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)



# Function that can be interact with the model
def chat():
    print("Hi, How can i help you ?")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = lables[results_index]
        if results[results_index] > 0.8:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            sleep(3)
            Bot = random.choice(responses)
            print(Bot)
        else:
            print("I don't understand!")
chat()
    




