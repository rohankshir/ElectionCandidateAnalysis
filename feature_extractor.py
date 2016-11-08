#! /Users/rohan/miniconda/bin/python
# Expects lines in the format '[label]\t[Sentence]'

import fileinput
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.corpus import stopwords
import sys
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

nltk.data.path.append('./nltk_data/')

import sklearn.linear_model as linear_model
from time import time
import numpy as np
np.random.seed(1337)  # for reproducibility
import string
from sklearn.metrics import confusion_matrix
from scipy import sparse
import dill as pickle


exclude = set(string.punctuation)

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels]+[5]) # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print "    " + empty_cell,
    for label in labels: 
        print "%{0}s".format(columnwidth) % label,
    print
    # Print rows
    for i, label1 in enumerate(labels):
        print "    %{0}s".format(columnwidth) % label1,
        for j in range(len(labels)): 
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print cell,
        print
        
def clean_token(token):
    if "--" in token:
        return "-"
    return token

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        
    def __call__(self, doc):
        ret = []
        for  t in doc.split():
            try:
                ret.append(self.wnl.lemmatize(clean_token(t.decode())))
            except NameError as e:
                pprint(e)
                ret.append(t)
        return ret

class POSTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        
    def __call__(self, doc):
        ret = []
        for  word,tag in nltk.pos_tag(nltk.word_tokenize(doc)):
                ret.append(tag)
        return ret


class StemmingTokenizer(object):
    def __init__(self):
        self.stem = PorterStemmer()
        
    def __call__(self, doc):
        ret = []
        for  t in nltk.word_tokenize(doc):
            try:
                word = self.stem.stem(clean_token(t.decode()))
                has_punct = any(elem in exclude for elem in word)
                if has_punct:
                    continue
                ret.append(word)
            except NameError as e:
                pprint(e)
                ret.append(t)
        return ret

def get_lexical_vectorizer(vocab=None):
    return CountVectorizer(min_df=1,
                           tokenizer=StemmingTokenizer(),
                           preprocessor=preprocess_sentence,
                           ngram_range=(1,1),
                           vocabulary=vocab,
                           stop_words='english')

def get_pos_vectorizer(vocab=None):
    return CountVectorizer(min_df=1,
                    tokenizer=POSTokenizer(),
                    preprocessor=preprocess_sentence,
                    ngram_range=(1,1),
                    vocabulary=vocab,
                    stop_words='english')

    
def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)

def remove_bracket(sentence):
    ret = ""
    in_bracket = False
    for c in sentence:
        if c == '[':
            in_bracket = True
        if not in_bracket:
            ret += c
        if c == ']':
            in_bracket = False
    return ret.strip()

def remove_annotations(sentence):
    if "D. TRUMP" in sentence:
        return sentence.split(':')[1].strip()
    return sentence
    
def preprocess_sentence(sentence):
    ret = remove_bracket(sentence)
    ret = remove_annotations(ret)
    return ret.lower()

def print_features(feats, feature_index):
    feats = feats.flatten()
    ret = []
    for i,hot in enumerate(feats):
        if hot and i in feature_index:
            ret.append(feature_index[i])
    return ret

def get_arbitary_features(sentences):
    features = np.zeros((len(sentences), 1))
    for i,sentence in enumerate(sentences):
        sentence = preprocess_sentence(sentence)
        features[i,0] = len(sentence.split())

    return features

class ArbitraryFeaturesVectorizer:
    def transform(self, sentences):
        return get_arbitary_features(sentences)

def main():

    reload(sys)
    sys.setdefaultencoding('utf-8')



    param_grid = [
        {'C': [.125, .25, .5, 1, 10, 100, 1000]},
        { 'penalty': ('l1','l2')}
    ]

    svm_param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    lines = [line for line in fileinput.input()]

    sentences = map(lambda x: x.split('\t')[1], lines)
    Y =  map(lambda x: int(x.split('\t')[0]), lines)

    vectorizer = get_lexical_vectorizer()

    pos_vectorizer = get_pos_vectorizer()


    pipeline = Pipeline([
        ('vect', vectorizer),
        ('clf', SGDClassifier()),
    ])

    # pprint(parameters)
    # t0 = time()
    # grid_search.fit(sentences, Y)
    # print("done in %0.3fs" % (time() - t0))
    # print()

    # print("Best score: %0.3f" % grid_search.best_score_)

    X = vectorizer.fit_transform(sentences)
    X = X.toarray()
    X_pos = pos_vectorizer.fit_transform(sentences)
    X_pos = X_pos.toarray()
    X_other_features = get_arbitary_features(sentences)
    X = np.hstack((X, X_pos, X_other_features))

    num_samples = len(Y)
    num_train = int(num_samples * .8)
    print "Num training: %d" % num_train
    X_train = X[0:num_train]
    Y_train = Y[0:num_train]
    X_test  = X[num_train:]
    Y_test = Y[num_train:]
    analyze = vectorizer.build_analyzer()

    for sentence in sentences[0:10]:
        print preprocess_sentence(sentence)
        print analyze(sentence)
        print "LemmaTokenizer" +  str(LemmaTokenizer()(sentence))
        print StemmingTokenizer()(sentence)

    # tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    # tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
    # chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)

    logistic = linear_model.LogisticRegression(C=.5, class_weight=None, dual=False,
                                               fit_intercept=True, intercept_scaling=1, max_iter=100,
                                               multi_class='ovr', penalty='l2', random_state=None,
                                               solver='liblinear', tol=0.0001, verbose=0)

    # grid_search = GridSearchCV(SVC(), svm_param_grid, n_jobs=-1, verbose=1)
    # grid_search.fit(X_train, Y_train)
    # print grid_search.score(X_test, Y_test)
    # best_parameters = grid_search.best_estimator_.get_params()
    # print best_parameters

    # grid_search = GridSearchCV(logistic, param_grid, n_jobs=-1, verbose=1)
    # grid_search.fit(X_train, Y_train)
    # print grid_search.score(X_test, Y_test)
    # best_parameters = grid_search.best_estimator_.get_params()
    # print best_parameters

    print logistic.fit(X_train,Y_train).score(X_test,Y_test)

    show_most_informative_features(vectorizer, logistic, 25)

    feature_vectorizers = [vectorizer, pos_vectorizer, ArbitraryFeaturesVectorizer()]
    with open('lexical_vocab.pkl','wb') as f:
        pickle.dump(vectorizer.vocabulary_, f)

    with open('pos_vocab.pkl','wb') as f:
        pickle.dump(pos_vectorizer.vocabulary_, f)
        
    with open('classifier.pkl','wb') as f:
        pickle.dump(logistic, f)

    from predictor import Predictor
    predictor = Predictor('classifier.pkl', 'lexical_vocab.pkl', 'pos_vocab.pkl')

    num_errors = 0

    # feature_names = vectorizer.vocabulary_
    # feature_index = {v: k for k, v in feature_names.items()}
    
    y_pred = []
    for (i,x) in enumerate(X_test):
        y_hat = predictor.predict_x(x)
        if y_hat != Y_test[i]:
            num_errors += 1
            # print "\n\nError predicting sentence: " + sentences[i + num_train]
            # print print_features(x, feature_index)
            # print "Label:{} Prediction: {}".format(Y_test[i], y_hat)
    error_rate = float(num_errors) / len(Y_test)
    print "Accuracy : " + str(1 - error_rate)
if __name__ == "__main__":
    main()
