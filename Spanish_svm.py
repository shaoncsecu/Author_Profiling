#!/usr/bin/env python
import argparse
from os import listdir
import os.path
import re
import nltk
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
import sys
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC
import time
from nltk.stem.porter import *
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import cross_val_score
import re
import numpy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
import warnings
from stop_words import get_stop_words
import random
from collections import Counter
import scikitplot as skplt
import matplotlib.pyplot as plt
random.seed(1337)

# Filters all warnings - specially to ignore warnings due to 0.0 in precision and recall and f-measures
warnings.filterwarnings("ignore")


def find_files(directory):
    paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            paths.append(filepath)
    return [path for path in paths if path.endswith(".xml")]


def find_truth(directory):
    paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            paths.append(filepath)
    return [path for path in paths if path.endswith(".txt")]


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train", required=True, type=str, help="Train folder")
    parser.add_argument("-te", "--test", required=True, type=str, help="Test folder")

    args = parser.parse_args()
    return args

# a dummy function that just returns its input
def identity(x):
    return x


# preprocessor for test documents
def process(x):

    # Removes the @username
    x = [w.replace('@username', '') for w in x]

    # Removes the hyperlinks
    x = [re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',w) for w in x]

    # Making everything lowercase
    x = [word.lower() for word in x]

    # This is the basic code for a single list - won't work for list of list
    # x = [word for word in x if word not in stopwords.words('dutch')]

    # Removing stopwords (specially for test documents)
    # This has to be done this way because it's a list of list - replacement fo above code fragment
    stop_words = get_stop_words('spanish')
    for stop in stop_words:
        x = [w.replace(" "+stop+" ", '') for w in x]

    # if you want to see the stop word list
    # for word in stop_words:
    #     print(word)

    return x


# preprocessor for classifier
def preProcess(x):
    # switches:
    splitter = True
    remove_username = True

    # simple tokenizer:
    if splitter:
        x = x.split()

    return x

# Apply Stemmer on Documents (I can't figure out how to pass it in Pipeline so done it separately)
def stem_documents(documents):
    # Apply Porter's stemming (pre-processor)
    stemmed_documents = []
    for doc in documents:
        stemmed_documents.append(apply_stemmer(doc))

    return stemmed_documents


# we are using NLTK stemmer to stem multiple words into root
def apply_stemmer(doc):
    stemmer = PorterStemmer()

    roots = [stemmer.stem(plural) for plural in doc]

    return roots


# NLTK POS Tagger
def tokenize_pos(tokens):
    return [token+"_POS-"+tag for token, tag in nltk.pos_tag(tokens)]


# Using NLTK lemmatizer
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


# Read the training and Testing Corpus
def read_corpus(corpus_folder, use_gender):
    print("Loading {0} data...".format(corpus_folder.split("/")[0]))
    # make empty lists
    corpus = []
    corpus_truth = []
    documents = []
    labels = []
    id = []

    # make tweet counter
    tweet_counter = 0

    # load train data
    corpus_files = find_files(corpus_folder)
    for fl in corpus_files:
        author_tweets = 0
        string = ""
        tree = ET.parse(fl)
        root = tree.getroot()
        author = root.attrib['id']
        for document in root.findall('document'):
            string = string +" "+ "".join(document.text.rstrip())
            tweet_counter += 1
            author_tweets += 1

        if use_gender:
            print("Author: {1} \t tweets: {0}".format(author_tweets, author))

        tuple = author, string
        corpus.append(tuple)
        id.append(author)

    # print amount of tweets loaded (only once for gender)
    if use_gender:
        print("\nTotal number of tweets: {0}".format(tweet_counter))

    # load train truth labels
    corpus_truth_file = find_truth(corpus_folder)
    with open(corpus_truth_file[0], "r") as corpus_truth_fl:
        for line in corpus_truth_fl:
            line = line.rstrip()
            corpus_truth.append(line)

    for corpus_line in corpus:
        corpus_author = corpus_line[0]
        corpus_documents = corpus_line[1]
        for truth_line in corpus_truth:
            truth_line = truth_line.split(":::")
            truth_author = truth_line[0]
            gender = truth_line[1]
            age = truth_line[2]

            if use_gender:
                if corpus_author == truth_author:
                    documents.append(corpus_documents)
                    labels.append(gender)

            else:
                if corpus_author == truth_author:
                    documents.append(corpus_documents)
                    labels.append(age)


    return documents, labels, id


# def read_testing_corpus(corpus_folder):
#     print("Loading {0} data...".format(corpus_folder.split("/")[0]))
#     # make empty lists
#     documents = []
#     id = []
#
#     # load train data
#     corpus_files = find_files(corpus_folder)
#     for fl in corpus_files:
#         string = ""
#         tree = ET.parse(fl)
#         root = tree.getroot()
#         author = root.attrib['id']
#         for document in root.findall('document'):
#             string = string +" "+ "".join(document.text.rstrip())
#
#         id.append(author)
#         documents.append(string)
#
#     return documents, id


def baseline_func():
    # baseline with tokenized tokens
    baseline_vec = CountVectorizer(preprocessor = preProcess, tokenizer = identity, ngram_range=(1, 1), analyzer='word')

    return baseline_vec


# decide on TF-IDF vectorization for feature
def tf_idf_func_gender():

    stop_list = get_stop_words('spanish')

    # we use a dummy function as tokenizer

    tfidf_vec = TfidfVectorizer(preprocessor = preProcess, tokenizer = identity, stop_words = stop_list, ngram_range=(1, 2))

    count_vec = CountVectorizer(analyzer='char', ngram_range=(3, 5))

    vec = FeatureUnion([("tfidf", tfidf_vec), ('count', count_vec)])

    return vec


# Modified TF-IDF vectorization for features: Uses pre-processing
def tf_idf_func_age():

    stop_list = get_stop_words('spanish')

    # Using Length Vectorizer combined with Tf-Idf and Count Vectorizer (**warning it will take so much time for Linear SVM)
    tfidf_vec_word = TfidfVectorizer(stop_words=stop_list, preprocessor = preProcess,
                               tokenizer = identity, analyzer='word', ngram_range=(1, 2))

    tfidf_vec_char = TfidfVectorizer(analyzer='char', ngram_range=(3, 5))

    vec = FeatureUnion([("tfidf_word", tfidf_vec_word), ('tfidf_char', tfidf_vec_char)])

    return vec


# Using a Linear Kernel
# SVM Classifier: the value of boolean arg - use_gender decides on Gender(True) or Age(False) classification
def SVM_Linear(trainDoc, trainClass, testDoc, testClass, use_gender):

    # Based on the use_gender decide what Vectorizer function to use
    if use_gender:
        vec = tf_idf_func_gender()
    else:
        vec = tf_idf_func_age()


    # combine the vectorizer with a SVM classifier
    classifier = Pipeline( [('vec', vec),
                            ('cls', SVC(kernel='linear', gamma=0.9, C=1.0))] )

    t0 = time.time()
    # Here trainDoc are the documents from training set and trainClass is the class labels for those documents
    classifier.fit(trainDoc, trainClass)

    train_time = time.time() - t0

    t1 = time.time()
    # Use the classifier to predict the class for all the documents in the test set testDoc
    # Save those output class labels in testGuess
    testGuess = classifier.predict(testDoc)
    print()

    # feature name extraction:
    print("Features")
    #print(classifier.named_steps['vec']) # print CountVectorizer settings
    # print()
    #print(classifier.named_steps['cls']) # print TfidfVectorizer settings
    feature_names = classifier.named_steps['vec'].transformer_list[0][1].get_feature_names()
    feature_names = feature_names + classifier.named_steps['vec'].transformer_list[1][1].get_feature_names()
    print("Amount of features used: {0}".format(len(feature_names)))
    print()

    # count most common features
    feature_cnt = Counter()
    for feature in feature_names:
        feature_cnt[feature] += 1
    print("Most common features:")
    print(feature_cnt.most_common(50))
    print()

    print("Random selection of features:")
    random.shuffle(feature_names)
    print(feature_names[:50])

    # calculate test time
    test_time = time.time() - t1

    # Just to know the output type
    classType = "AGE"
    if use_gender:
        classType = "Gender[M/F]"

    print("\n########### Linear SVM Classifier For ", classType, " in Spanish ###########\n")

    # Call to function(s) to show the results for development set ^_^
    print("\n==>Measures on the Test set:\n")
    title = "Model - Spanish ("+classType+")"
    calculate_measures(classifier, testClass, testGuess, title)

    # Doing cross validation on the Training set (test set is separated to do normal measure)
    print("\n==>Cross-validation on the Training set:\n")
    cross_validation(classifier, trainDoc, trainClass)

    print("\nTraining Time: ", train_time)
    print("Testing Time: ", test_time)

    return testGuess


# Using a Linear Kernel with Basic Settings
def SVM_baseline(trainDoc, trainClass, testDoc, testClass, use_gender):

    vec = baseline_func()

    # combine the vectorizer with a SVM classifier
    classifier = Pipeline( [('vec', vec),
                            ('cls', SVC(kernel='linear'))] )

    t0 = time.time()
    classifier.fit(trainDoc, trainClass)

    train_time = time.time() - t0
    t1 = time.time()

    # For Baseline we don't need this
    testGuess = classifier.predict(testDoc)

    test_time = time.time() - t1

    # Just to know the output type
    classType = "AGE"
    if use_gender:
        classType = "Gender[M/F]"

    print("\n########### Baseline SVM Classifier For ", classType, " in Spanish ###########\n")

    # Call to function(s) to show the results for development set ^_^
    print("\n==>Measures on the Test set:\n")
    title = "Baseline - Spanish ("+classType+")"
    calculate_measures(classifier, testClass, testGuess, title)

    # Doing cross validation on the Training set (test set is separated to do normal measure)
    print("\n==>Cross-validation on the Training set:\n")
    cross_validation(classifier, trainDoc, trainClass)

    print("\nTraining Time: ", train_time)
    print("Testing Time: ", test_time)


# for calculating different scores
def calculate_measures(classifier, testClass, testGuess, title):

    # Compare the accuracy of the output (Yguess) with the class labels of the original test set (Ytest)
    print("Accuracy = "+str(accuracy_score(testClass, testGuess)))

    # Report on the precision, recall, f1-score of the output (Yguess) with the class labels of the original test set (Ytest)
    print(classification_report(testClass, testGuess, labels=classifier.classes_, target_names=None, sample_weight=None, digits=3))

    # Showing the Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(testClass, testGuess, labels=classifier.classes_)
    print(classifier.classes_)
    print(cm)
    print()

    # Drawing Confusion Matrix
    skplt.metrics.plot_confusion_matrix(testClass, testGuess, normalize=True)
    tick_marks = numpy.arange(len(classifier.classes_))
    plt.xticks(tick_marks, classifier.classes_, rotation=45)
    plt.yticks(tick_marks, classifier.classes_)
    plt.title("Normalized Confusion Matrix: "+title)
    plt.tight_layout()
    plt.show()


# This function cross validates the Training set into n folds
# We will not use the test set to cross validate because we want to compare the results
# of average cross validation score to the normal scores with test set
def cross_validation(classifier, trainDoc, trainClass):

    # Showing 3 fold cross validation score cv = no. of folds
    n_fold = 3
    scores = cross_val_score(classifier, trainDoc, trainClass, cv=n_fold, n_jobs=-1)

    print(n_fold,"-fold Cross Validation (Accuracy):\n", scores)
    print("\nAccuracy (Mean - Cross Validation): %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    scores = cross_val_score(classifier, trainDoc, trainClass, cv=n_fold, scoring='f1_macro', n_jobs=-1)

    print("\nF1-macro (Mean - Cross Validation): %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    scores = cross_val_score(classifier, trainDoc, trainClass, cv=n_fold, scoring='precision_macro', n_jobs=-1)

    print("\nPrecision (Mean - Cross Validation): %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    scores = cross_val_score(classifier, trainDoc, trainClass, cv=n_fold, scoring='recall_macro', n_jobs=-1)

    print("\nRecall (Mean - Cross Validation): %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# value of use_gender is True for gender and False for age
def run_classifier(use_gender, args):
    # using runtime command arguments to get the folder path
    train_folder = args.train
    test_folder = args.test

    # load data for traning and testing
    trainDoc, trainClass, train_id = read_corpus(train_folder, use_gender)
    testDoc, testClass, test_id = read_corpus(test_folder, use_gender)

    # print amount of files loaded
    print("Train files: {0}, test files: {1}".format(len(trainDoc), len(testDoc)))

    # Running the Baseline Classifier
    SVM_baseline(trainDoc, trainClass, testDoc, testClass, use_gender)

    # Preprocessing test documents before feeding it to the classifier.
    if use_gender:
        trainDoc = process(trainDoc)
        testDoc = process(testDoc)


    # Calling the classifier function (Original Model - Linear SVM)
    testGuess = SVM_Linear(trainDoc, trainClass, testDoc, testClass, use_gender)

    # testGuess =  SVM_Linear(apply_stemmer(trainDoc), trainClass, apply_stemmer(testDoc), use_gender)     #with stammer

    return test_id, testGuess


def main():

    # create args parser
    args = create_arg_parser()

    # calling the classifier for gender and age
    id_gen, guess_gen = run_classifier(True, args)    # Predict Gender (True)
    print("\n")
    id_age, guess_age = run_classifier(False, args)   # Predict Age (False)

    # combining the guess for age and gender with author id to creat the truth.txt file in the test folder
    out = []
    for id1, id2, gen, age in zip(id_gen, id_age, guess_gen, guess_age):
        # out_line = id1+":::"+gen+":::"+age+":::"+id2                      # if you want you can match id1 with id2
        out_line = id1+":::"+gen+":::"+age
        out.append(out_line)

    # creating the truth.txt file in the test/english folder
    with open(args.test+"truth.txt", 'w') as f:
        f.write('\n'.join(out))

if __name__ == '__main__':
    main()
