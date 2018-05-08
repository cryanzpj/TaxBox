import numpy as np
import nltk
import pickle
import random
import json
import tflearn
import tensorflow as tf
from nltk.stem.lancaster import LancasterStemmer
from taxcode_tfidf_search_script import *


def clean_up_sentence(sentence):
    # tokenize the pattern
    stemmer = LancasterStemmer()
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    return (np.array(bag))


ERROR_THRESHOLD = 0.25


def classify(sentence, model, words, classes):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list


def response(sentence, intents, model, words, classes):
    results = classify(sentence, model, words, classes)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # a random response from the intent
                    # return print(random.choice(i['responses']))
                    return random.choice(i['responses'])

            results.pop(0)


def full_response(sentence, top_n, bin, thres=0.2):
    answer = response(sentence,bin['intents'],bin['inten_model'], bin['intent_words'], bin['intent_classes'])

    if answer == "Tax code is":
        return_code = 0
        temp = query_wrapper(sentence, cosine_sim_threshold=thres, bin = bin,top = top_n)
        if len(temp) == 0:
            final_code = 1
            final = "No tax code found"
        else:
            #temp_values = temp['title'].values
            final_code = 0
            final = temp

    elif answer == "Here is what we found in the tax code:":
        return_code = 1
        temp = query_wrapper(sentence, cosine_sim_threshold=thres, bin = bin,top=top_n)
        if len(temp) == 0:
            final_code = 1
            final = "We have not found any section in tax code related to your question"
        else:
            #temp_values = temp['title'].values + " " + temp['text'].values
            final_code = 0
            final = temp

    else:
        final_code = 0
        return_code = 2
        final = answer

    return return_code,final_code,final


def load_tf_model():
    def load_tf():
        tf.reset_default_graph()
        net = tflearn.input_data(shape=[None, 48])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 6, activation='softmax')
        net = tflearn.regression(net)
        model = tflearn.DNN(net)
        model.load("./Bin/tf_model/model.tflearn")
        return model

    with open("./Bin/intents.json") as json_data:
        intents = json.load(json_data)
    model = load_tf()
    data = pickle.load(open("./Bin/training_data", "rb"))
    words = data['words']
    classes = data['classes']

    return model, words, classes, intents
