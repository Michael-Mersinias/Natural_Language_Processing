import gc
import math
import nltk
import re
import string
import pandas as pd
import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros
# from pattern.en import lemma
from nltk.stem import WordNetLemmatizer
from collections import Counter
from scipy import sparse
from scipy import spatial
from sklearn import preprocessing
from sklearn import metrics
from xgboost import XGBClassifier

import json
import os
import gensim
import time

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

import tensorflow as tf

import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Conv1D, MaxPooling1D, \
    GlobalMaxPooling1D, Activation, Flatten, TimeDistributed, RepeatVector, GRU

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


# Functions to reset runtime and clear memory.

def clear_global_memory():
    try:
        for name in globals():
            del globals()[name]
    except:
        pass

    gc.collect()


def clear_local_memory():
    try:
        for name in locals():
            del locals()[name]
    except:
        pass

    gc.collect()


def clear_memory_variable(name):
    try:
        del globals()[name]
    except KeyError:
        pass

    try:
        del locals()[name]
    except KeyError:
        pass

    gc.collect()


# Data Preprocessing

# Removing all instances without a label or where the character length of the text is too short.

def clean_data(data):
    data = data.replace("\t", "\n")

    text_tmp = data.body_text.values
    title_tmp = data.title_text.values
    label_tmp = data.label.values

    indexes_to_drop = []

    for i in range(0, (len(text_tmp))):
        if str(label_tmp[i]) != '1' and str(label_tmp[i]) != '0':
            indexes_to_drop.append(i)
        if ((len(str(text_tmp[i])) < 100) or (len(str(title_tmp[i])) < 5) or (
                isinstance(text_tmp[i], str) == False) or (isinstance(title_tmp[i], str) == False)):
            indexes_to_drop.append(i)

    indexes_to_keep = set(range(data.shape[0])) - set(indexes_to_drop)
    data = data.take(list(indexes_to_keep))

    data = data.dropna()
    data = data.reset_index(drop=True)

    return data


# Checking if there are null or invalid instances in our dataset.

def check_data(data):
    if (data.isnull().sum().sum()) == 0:  # or ((len(text_tmp[i]) >= 100) or (len(title_tmp[i]) >= 5)):
        print("\nOur dataset has no missing data\n")
    else:
        print("\nOur dataset needs to be checked for missing data\n")


# Stopword removal, lemmatization.

def preprocess(data, stopwords=nltk.corpus.stopwords.words("english"), wnl=WordNetLemmatizer(), tokenized=False):
    # data = "".join([word.lower() for word in data if word not in string.punctuation])
    data = re.split("\W+", data)
    data = [(wnl.lemmatize(str(word), 'a')).lower() for word in data if
            ((word.lower() not in stopwords) and (word.lower() not in string.punctuation))]

    if tokenized == False:
        data = ' '.join(data)

    return data


# Combining all of the above functions to preprocess the text.

def prepare_data(data):
    data.columns = ["title_text", "body_text", "label"]

    data = clean_data(data)
    check_data(data)

    data['title_text'] = data['title_text'].apply(preprocess)
    data['body_text'] = data['body_text'].apply(preprocess)

    data = clean_data(data)
    check_data(data)

    return data


# K-fold cross validation.

def cross_val_split(X, y, k_folds):
    train_index_list = []
    test_index_list = []

    kf = StratifiedKFold(n_splits=k_folds)
    kf.get_n_splits(X, y)

    for train_index, test_index in kf.split(X, y):
        train_index_list.append(train_index)
        test_index_list.append(test_index)

    return train_index_list, test_index_list


# Term Frequency (tf) vectorization.

def count_vectorizer(x_train, x_test):
    vectorizer = CountVectorizer()
    vectorizer.fit(x_train)
    x_train = vectorizer.transform(x_train)
    x_test = vectorizer.transform(x_test)

    return x_train, x_test


# Term Frequency - Inverse Document Frequency (tf-idf) vectorization.

def tfidf_vectorizer(x_train, x_test):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(x_train)
    x_train = vectorizer.transform(x_train)
    x_test = vectorizer.transform(x_test)

    return x_train, x_test


# Class Label Frequency Distance (CLFD) vectorization (3 functions)


# 1st function: Creates the three cltf vectors: "terms" and "occ" (real_words, fake_words) as well as "total" (total_real_counts, total_fake_counts)

def get_term_custom_score(x_train, y_train):
    score = 0
    word_list = []
    score_list = []

    x_train_real = x_train.loc[y_train == 0]
    x_train_fake = x_train.loc[y_train == 1]

    y_train_real = y_train.loc[y_train == 0]
    y_train_fake = y_train.loc[y_train == 1]

    tst_real = x_train_real.str.cat(sep=' ')
    tst_fake = x_train_fake.str.cat(sep=' ')

    tokenized_real = re.split('\W+', tst_real)
    tokenized_fake = re.split('\W+', tst_fake)

    real_words = Counter(tokenized_real)
    fake_words = Counter(tokenized_fake)

    total_real_counts = len(tokenized_real)
    total_fake_counts = len(tokenized_fake)

    return real_words, total_real_counts, fake_words, total_fake_counts


# 2nd function: Utilizes the three cltf vectors to create the clfr scores for each class (score_real, score_fake) as well as the clfd score for each word (score)

def custom_vectorizer(vectorizer, x_train, real_words, total_real_counts, fake_words, total_fake_counts):
    ze_words = vectorizer.vocabulary_

    word_id = vectorizer.get_feature_names()

    score_array = []

    for i in range(0, len(word_id)):
        word = word_id[i]

        real_freq = real_words[word]

        fake_freq = fake_words[word]

        score_real = math.log(1 + (((1 + real_freq) / total_real_counts) / ((1 + fake_freq) / total_fake_counts)))
        score_fake = math.log(1 + (((1 + fake_freq) / total_fake_counts) / ((1 + real_freq) / total_real_counts)))

        score = max(score_real, score_fake) - min(score_real, score_fake)

        score_array.append(score)

    # Multiplying the clfd array with either tf vectorizer or tfidf vectorizer, to generate tf-clfd or tfidf-clfd vectors.

    custom_array1 = x_train.multiply(sparse.csr_matrix(score_array))

    # Multiplying the clfd array with a b-tf vectorizer (any term with a greater occurrence than 1, becomes 1), to generate b-clfd vectors.

    x_train[x_train >= 1] = 1
    custom_array2 = x_train.multiply(sparse.csr_matrix(score_array))

    return custom_array1, custom_array2


# Depending on the choice between b-clfd, tf-clfd and tfidf-clfd, we choose one of the following three functions.

def clfd_vectorizer(x_train, x_test, y_train):
    real_words, total_real_counts, fake_words, total_fake_counts = get_term_custom_score(x_train, y_train)

    vectorizer = CountVectorizer(preprocessor=None, lowercase=False)  # CountVectorizer TfidfVectorizer
    vectorizer.fit(x_train)
    x_train = vectorizer.transform(x_train)
    x_test = vectorizer.transform(x_test)

    custom_array1, custom_array2 = custom_vectorizer(vectorizer, x_train, real_words, total_real_counts, fake_words,
                                                     total_fake_counts)

    test_array1, test_array2 = custom_vectorizer(vectorizer, x_test, real_words, total_real_counts, fake_words,
                                                 total_fake_counts)

    custom_array1 = sparse.csr_matrix(custom_array1)
    test_array1 = sparse.csr_matrix(test_array1)
    custom_array2 = sparse.csr_matrix(custom_array2)
    test_array2 = sparse.csr_matrix(test_array2)

    return custom_array2, test_array2


def tf_clfd_vectorizer(x_train, x_test, y_train):
    real_words, total_real_counts, fake_words, total_fake_counts = get_term_custom_score(x_train, y_train)

    vectorizer = CountVectorizer(preprocessor=None, lowercase=False)  # CountVectorizer TfidfVectorizer
    vectorizer.fit(x_train)
    x_train = vectorizer.transform(x_train)
    x_test = vectorizer.transform(x_test)

    custom_array1, custom_array2 = custom_vectorizer(vectorizer, x_train, real_words, total_real_counts, fake_words,
                                                     total_fake_counts)

    test_array1, test_array2 = custom_vectorizer(vectorizer, x_test, real_words, total_real_counts, fake_words,
                                                 total_fake_counts)

    custom_array1 = sparse.csr_matrix(custom_array1)
    test_array1 = sparse.csr_matrix(test_array1)
    custom_array2 = sparse.csr_matrix(custom_array2)
    test_array2 = sparse.csr_matrix(test_array2)

    return custom_array1, test_array1


def tf_idf_clfd_vectorizer(x_train, x_test, y_train):
    real_words, total_real_counts, fake_words, total_fake_counts = get_term_custom_score(x_train, y_train)

    vectorizer = TfidfVectorizer(preprocessor=None, lowercase=False)  # CountVectorizer TfidfVectorizer
    vectorizer.fit(x_train)
    x_train = vectorizer.transform(x_train)
    x_test = vectorizer.transform(x_test)

    custom_array1, custom_array2 = custom_vectorizer(vectorizer, x_train, real_words, total_real_counts, fake_words,
                                                     total_fake_counts)

    test_array1, test_array2 = custom_vectorizer(vectorizer, x_test, real_words, total_real_counts, fake_words,
                                                 total_fake_counts)

    custom_array1 = sparse.csr_matrix(custom_array1)
    test_array1 = sparse.csr_matrix(test_array1)
    custom_array2 = sparse.csr_matrix(custom_array2)
    test_array2 = sparse.csr_matrix(test_array2)

    return custom_array1, test_array1


# The metrics of performance: Accuracy, Precision, Recall, F-1 Score are the ones we focus on.

def metric_function(y_test, y_pred_prob, y_pred):
    recall = (metrics.recall_score(y_test, y_pred)).mean() * 100
    precision = (metrics.precision_score(y_test, y_pred)).mean() * 100
    fscore = metrics.f1_score(y_test, y_pred) * 100
    accuracy = metrics.accuracy_score(y_test, y_pred) * 100
    # logloss = metrics.log_loss(y_test, y_pred_prob[:, 1])
    # roc_auc = metrics.roc_auc_score(y_test, y_pred_prob[:, 1])
    # conf_matrix = metrics.confusion_matrix(y_test, y_pred)

    # conf_matrix[0][0] = int(conf_matrix[0][0])
    # conf_matrix[0][1] = int(conf_matrix[0][1])
    # conf_matrix[1][0] = int(conf_matrix[1][0])
    # conf_matrix[1][1] = int(conf_matrix[1][1])

    return accuracy, precision, recall, fscore  # , logloss, roc_auc, conf_matrix


def classification(classifier, x_train, x_test, y_train, y_test):
    res_scores = []
    res_preds = []

    model = classifier.fit(x_train, y_train)
    y_pred_prob = model.predict(x_test)
    y_pred = (model.predict(x_test) > 0.5)

    accuracy, precision, recall, fscore = metric_function(y_test, y_pred_prob, y_pred)

    report = [accuracy, precision, recall, fscore]
    preds = [y_pred_prob, y_pred]

    res_scores.append(report)
    res_preds.append(preds)

    return res_scores, res_preds


def main():
    # Menu

    choice = -1

    print('\n')
    print('1.US Elections Dataset (size=6335)')
    print('2.Kaggle Dataset (size=20718)')
    print('3.Custom Dataset (size=38010)')

    while choice < 1 or choice > 3:
        print('\n')
        choice = int(input('Choose the dataset: '))

    # Change the path in case the datasets are stored in a different google drive path.

    if choice == 1:
        dataset_path = 'gdrive/My Drive/Colab Notebooks/fake_or_real_news.csv'
        dataset_title = 'US Elections Dataset (size=6335)'
    elif choice == 2:
        dataset_path = 'gdrive/My Drive/Colab Notebooks/data_train.csv'
        dataset_title = 'Kaggle Dataset (size=20718)'
    elif choice == 3:
        dataset_path = 'gdrive/My Drive/Colab Notebooks/full_dataset.csv'
        dataset_title = 'Custom Dataset (size=38010)'
    else:
        print('Error')

    times = []
    final_report = []

    # Specific preprocessing for each dataset.

    if dataset_path == 'gdrive/My Drive/Colab Notebooks/data_train.csv':
        dataset = pd.read_csv("data/data_train.csv", encoding="ISO-8859-1",
                              index_col=0)
        dataset = dataset[['title', 'text', 'label']]
        models = [LogisticRegression(), RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1),
                  MultinomialNB(alpha=1), XGBClassifier(n_estimators=200)]  # AdaBoostClassifier(n_estimators=200)
    elif dataset_path == 'gdrive/My Drive/Colab Notebooks/fake_or_real_news.csv':
        dataset = pd.read_csv("data/fake_or_real_news.csv", encoding="utf8",
                              index_col=0)
        dataset['label'] = dataset['label'].replace('FAKE', 1)
        dataset['label'] = dataset['label'].replace('REAL', 0)
        models = [LogisticRegression(), RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1),
                  MultinomialNB(alpha=1), XGBClassifier(n_estimators=150)]  # AdaBoostClassifier(n_estimators=150)
    else:
        dataset = pd.read_csv("data/full_dataset.csv", encoding="utf8",
                              index_col=0)
        models = [LogisticRegression(), RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1),
                  MultinomialNB(alpha=0.0001), XGBClassifier(n_estimators=100)]  # AdaBoostClassifier(n_estimators=100)

    feature_titles = ['Term Frequency', 'Term Frequency - Inverse Document Frequency', 'Class Label Frequency Distance',
                      'Term Frequency - Class Label Frequency Distance',
                      'Term Frequency - Inverse Document Frequency - Class Label Frequency Distance']
    model_titles = ['Logistic Regression', 'Random Forest', 'Multinomial Naive Bayes', 'Gradient Boosting', 'LSTM',
                    'Bi-LSTM', 'CNN+LSTM', 'Stacked LSTM', 'Hybrid model']  # 'Adaptive Boost'

    choice1 = -1

    print('\n')
    for i in range(0, len(model_titles)):
        print(str(i + 1) + '.' + model_titles[i])

    while choice1 < 1 or choice1 > 9:
        print('\n')
        choice1 = int(input('Choose the machine learning method: '))

    model_title = model_titles[choice1 - 1]

    if choice1 == 9:  # Hybrid Model

        y_pred_prob = 0
        y_pred = 0

        feature_title = 'clfd + word embeddings'

        dataset = prepare_data(dataset)
        print(dataset.head())
        dataset.label = dataset.label.astype(int)
        X = dataset['body_text']
        y = dataset['label']

        start_time = time.time()

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1000)

        # LOGISTIC REGRESSION

        x_train1, x_test1 = clfd_vectorizer(x_train, x_test, y_train)

        model = LogisticRegression()

        model = model.fit(x_train1, y_train)
        y_pred_prob1 = model.predict_proba(x_test1)
        y_pred1 = (model.predict(x_test1) > 0.5)

        # CNN+LSTM

        t = Tokenizer()
        t.fit_on_texts(X)
        vocabulary_size = len(t.word_index) + 1

        X = t.texts_to_sequences(X)

        y = dataset['label'].values

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1000)

        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        max_vocab_train = (max(max(doc) for doc in x_train))
        max_vocab_test = (max(max(doc) for doc in x_test))
        vocab_size = max(max_vocab_train, max_vocab_test) + 1

        max_length_train = max(len(doc) for doc in x_train)
        max_length_test = max(len(doc) for doc in x_test)
        max_length = max(max_length_train, max_length_test)

        batch_size = 32
        embedding_size = 128  # This needs to be 100 for pre-trained glove vectors

        max_features = vocab_size
        maxlen = max_length
        x_train = sequence.pad_sequences(x_train, maxlen=maxlen, padding='pre')
        x_test = sequence.pad_sequences(x_test, maxlen=maxlen, padding='pre')

        model1 = Sequential()
        model1.add(Embedding(max_features, embedding_size, input_length=maxlen))
        model1.add(Dropout(rate=0.2))
        model1.add(Conv1D(128, 3, activation='relu'))
        model1.add(MaxPooling1D(3))
        model1.add(Conv1D(128, 3, activation='relu'))
        model1.add(MaxPooling1D(3))
        model1.add(Conv1D(128, 3, activation='relu'))
        model1.add(MaxPooling1D(3))
        model1.add(LSTM(128))
        model1.add(Dense(128, activation='relu'))
        model1.add(Dense(1, activation='sigmoid'))

        if dataset_path == 'gdrive/My Drive/Colab Notebooks/data_train.csv':
            filepath = 'models/MID_CNN_checkpoint.hdf5'
        elif dataset_path == 'gdrive/My Drive/Colab Notebooks/fake_or_real_news.csv':
            filepath = 'models/ELECTIONS_CNN_checkpoint.hdf5'
        else:
            filepath = 'models/MERGED_CNN_checkpoint.hdf5'

        model1.load_weights(filepath)
        model1.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

        y_pred_prob = model1.predict(x_test)
        y_pred = (model1.predict(x_test) > 0.5)

        y_pred2 = y_pred.astype(int)  # int(y_pred)
        y_pred_prob2 = y_pred_prob.astype(float)  # float(y_pred_prob)
        y_test = y_test.astype(int)  # new

        # NEW - CALCULATE COEFFICIENT

        lr_co = 0
        cnn_co = 0

        for v in range(0, len(y_pred_prob)):
            lr_co = lr_co + y_pred_prob1[v][0]
            cnn_co = cnn_co + y_pred_prob2[v]

        print(lr_co, cnn_co)

        alpha_coefficient = 0.7  # lr_co/(lr_co+cnn_co)

        print('LR COEF: ', alpha_coefficient)

        for v in range(0, len(y_pred_prob)):
            y_pred_prob[v] = (y_pred_prob1[v][0] * alpha_coefficient + (1 - y_pred_prob2[v]) * (1 - alpha_coefficient))  # /2
            if y_pred_prob[v] > 0.5:
                y_pred[v] = 0
            else:
                y_pred[v] = 1

        y_pred_hybrid = []
        for v in range(0, len(y_pred)):
            y_pred_hybrid.append(y_pred[v])

        y_pred_hybrid = np.array(y_pred_hybrid)

        # rec = (metrics.recall_score(y_test, y_pred_hybrid))*100
        # prec = (metrics.precision_score(y_test, y_pred_hybrid))*100
        # f1 = metrics.f1_score(y_test, y_pred_hybrid)*100
        # acc = metrics.accuracy_score(y_test, y_pred_hybrid)*100

        acc, prec, rec, f1 = metric_function(y_test, y_pred_prob, y_pred)
        report = [acc, prec, rec, f1]

        final_report.append(report)

        # report = [acc, prec, rec, f1]

        # final_report.append(report)

        end_time = time.time()
        time2 = end_time - start_time

        samples = len(y_test)

    elif choice1 > 4:  # Deep Learning methods

        dataset = prepare_data(dataset)
        # if(dataset_path=='gdrive/My Drive/Colab Notebooks/data_train.csv'):
        #    dataset.label = dataset.label.astype(int)
        print(dataset.head())
        dataset.label = dataset.label.astype(int)
        X = dataset['body_text']
        y = dataset['label']

        feature_title = 'Word embeddings'

        start_time = time.time()

        t = Tokenizer()
        t.fit_on_texts(X)
        vocabulary_size = len(t.word_index) + 1

        X = t.texts_to_sequences(X)

        Y = dataset['label'].values

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1000)

        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        max_vocab_train = (max(max(doc) for doc in x_train))
        max_vocab_test = (max(max(doc) for doc in x_test))
        vocab_size = max(max_vocab_train, max_vocab_test) + 1

        max_length_train = max(len(doc) for doc in x_train)
        max_length_test = max(len(doc) for doc in x_test)
        max_length = max(max_length_train, max_length_test)

        batch_size = 32
        embedding_size = 128  # This needs to be 100 for pre-trained glove vectors

        max_features = vocab_size
        maxlen = max_length
        x_train = sequence.pad_sequences(x_train, maxlen=maxlen, padding='pre')
        x_test = sequence.pad_sequences(x_test, maxlen=maxlen, padding='pre')

        if choice1 == 5:  # LSTM
            model1 = Sequential()
            model1.add(Embedding(max_features, embedding_size, input_length=maxlen))
            model1.add(Dropout(0.4))
            model1.add((LSTM(64)))
            model1.add(Dense((64 * 4), activation='relu'))  # *8
            model1.add(Dense(1, activation='sigmoid'))

            if dataset_path == 'gdrive/My Drive/Colab Notebooks/data_train.csv':
                filepath = 'models/MID_test_LSTM_checkpoint.hdf5'
            elif dataset_path == 'gdrive/My Drive/Colab Notebooks/fake_or_real_news.csv':
                filepath = 'models/ELECTIONS_LSTM_checkpoint.hdf5'
            else:
                filepath = 'models/MERGED_LSTM_checkpoint.hdf5'

        elif choice1 == 6:  # Bi-LSTM
            model1 = Sequential()
            model1.add(Embedding(max_features, embedding_size, input_length=maxlen))
            model1.add(Dropout(0.4))
            model1.add(Bidirectional(LSTM(64)))
            model1.add(Dense((64 * 4), activation='relu'))
            model1.add(Dense(1, activation='sigmoid'))

            if dataset_path == 'gdrive/My Drive/Colab Notebooks/data_train.csv':
                filepath = 'models/MID_BILSTM_checkpoint.hdf5'
            elif dataset_path == 'gdrive/My Drive/Colab Notebooks/fake_or_real_news.csv':
                filepath = 'models/ELECTIONS_BILSTM_checkpoint.hdf5'
            else:
                filepath = 'models/MERGED_biLSTM_checkpoint.hdf5'
                # filepath='gdrive/My Drive/MERGED_BILSTM_checkpoint.hdf5'

        elif choice1 == 7:  # CNN+LSTM
            model1 = Sequential()
            model1.add(Embedding(max_features, embedding_size, input_length=maxlen))
            model1.add(Dropout(rate=0.2))
            model1.add(Conv1D(128, 3, activation='relu'))
            model1.add(MaxPooling1D(3))
            model1.add(Conv1D(128, 3, activation='relu'))
            model1.add(MaxPooling1D(3))
            model1.add(Conv1D(128, 3, activation='relu'))
            model1.add(MaxPooling1D(3))
            model1.add(LSTM(128))
            model1.add(Dense(128, activation='relu'))
            model1.add(Dense(1, activation='sigmoid'))

            if dataset_path == 'gdrive/My Drive/Colab Notebooks/data_train.csv':
                filepath = 'models/MID_CNN_checkpoint.hdf5'
            elif dataset_path == 'gdrive/My Drive/Colab Notebooks/fake_or_real_news.csv':
                filepath = 'models/ELECTIONS_CNN_checkpoint.hdf5'
            else:
                filepath = 'models/MERGED_CNN_checkpoint.hdf5'

        elif choice1 == 8:  # Mult.LSTM
            model1 = Sequential()
            model1.add(Embedding(vocab_size, 64, input_length=maxlen))
            model1.add(Dropout(0.4))
            model1.add(LSTM(64))
            model1.add(RepeatVector(maxlen))
            model1.add(LSTM(64))
            model1.add(Dense((64 * 4), activation='relu'))
            model1.add(Dense(1, activation='sigmoid'))

            if dataset_path == 'gdrive/My Drive/Colab Notebooks/data_train.csv':
                filepath = 'models/MID_STACKEDLSTM_checkpoint.hdf5'
            elif dataset_path == 'gdrive/My Drive/Colab Notebooks/fake_or_real_news.csv':
                filepath = 'models/ELECTIONS_STACKED_3_LSTM_checkpoint.hdf5'
            else:
                filepath = 'models/MERGED_STACKEDLSTM_checkpoint.hdf5'

        model1.load_weights(filepath)
        model1.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

        y_pred_prob = model1.predict(x_test)
        y_pred = (model1.predict(x_test) > 0.5)

        y_pred = y_pred.astype(int)
        y_pred_prob = y_pred_prob.astype(float)
        y_test = y_test.astype(int)

        # accuracy, precision, recall, fscore = metric_function(y_test, y_pred_prob, y_pred)

        # rec = (metrics.recall_score(y_test, y_pred))*100
        # prec = (metrics.precision_score(y_test, y_pred))*100
        # f1 = metrics.f1_score(y_test, y_pred)*100
        # acc = metrics.accuracy_score(y_test, y_pred)*100

        acc, prec, rec, f1 = metric_function(y_test, y_pred_prob, y_pred)
        report = [acc, prec, rec, f1]

        final_report.append(report)

        # report = [acc, prec, rec, f1]

        # final_report.append(report)

        end_time = time.time()
        time2 = end_time - start_time

        samples = len(y_test)

    elif choice1 < 9:  # Linear machine learning method of choice.

        model = models[choice1 - 1]

        choice2 = -1

        print('\n')
        for i in range(0, len(feature_titles)):
            print(str(i + 1) + '.' + feature_titles[i])

        while choice2 < 0 or choice2 > 5:
            print('\n')
            choice2 = int(input('Choose the feature extraction technique: '))

        input_type = choice2
        feature_title = feature_titles[choice2 - 1]

        if input_type == 0:  # Ignore this part
            dataset.columns = ['title_text', 'body_text', 'label']
            dataset = clean_data(dataset)
            # dataset = apply_extra_features(dataset)
            dataset.label = dataset.label.astype(int)
            y = dataset['label'].values
            X = dataset.drop(['label'], axis=1)
            X = (preprocessing.MinMaxScaler()).fit_transform(X)
            X = sparse.csr_matrix(X)
        else:  # Always here
            dataset = prepare_data(dataset)
            print(dataset.head())
            dataset.label = dataset.label.astype(int)
            X = dataset['body_text']
            y = dataset['label']

        # K-fold cross validation

        k_folds = 5
        train_index_list, test_index_list = cross_val_split(X, y, k_folds)

        for i in range(0, len(train_index_list)):

            print('K_fold ' + str(i))
            start_time = time.time()

            y_pred_prob = 0
            y_pred = 0

            x_train = X[train_index_list[i]]
            x_test = X[test_index_list[i]]
            y_train = y[train_index_list[i]]
            y_test = y[test_index_list[i]]

            if input_type == 1:
                x_train, x_test = count_vectorizer(x_train, x_test)
            elif input_type == 2:
                x_train, x_test = tfidf_vectorizer(x_train, x_test)
            elif input_type == 3:
                x_train, x_test = clfd_vectorizer(x_train, x_test, y_train)
            elif input_type == 4:
                x_train, x_test = tf_clfd_vectorizer(x_train, x_test, y_train)
            elif input_type == 5:
                x_train, x_test = tf_idf_clfd_vectorizer(x_train, x_test, y_train)
            elif input_type != 0:
                print('Error: Bad input type')

            res_scores, res_preds = classification(model, x_train, x_test, y_train, y_test)
            y_pred_prob = y_pred_prob + res_preds[0][0]
            y_pred = y_pred + res_preds[0][1]

            accuracy, precision, recall, fscore = metric_function(y_test, y_pred_prob, y_pred)
            report = [accuracy, precision, recall, fscore]

            final_report.append(report)

            end_time = time.time()
            times.append((end_time - start_time))

        samples = len(y_test) * k_folds

        acc = []
        prec = []
        rec = []
        f1 = []
        time2 = []

        for j in range(0, k_folds):
            acc.append(final_report[j][0])  # *100)
            prec.append(final_report[j][1])  # *100)
            rec.append(final_report[j][2])  # *100)
            f1.append(final_report[j][3])  # *100)
            time2.append(times[j])

    # final_acc = [np.round(np.min(acc), 2), np.round(np.mean(acc), 2), np.round(np.max(acc), 2)]
    # final_prec = [np.round(np.min(prec), 2), np.round(np.mean(prec), 2), np.round(np.max(prec), 2)]
    # final_rec = [np.round(np.min(rec), 2), np.round(np.mean(rec), 2), np.round(np.max(rec), 2)]
    # final_f1 = [np.round(np.min(f1), 2), np.round(np.mean(f1), 2), np.round(np.max(f1), 2)]
    # final_time = [np.round(np.min(time2), 2), np.round(np.mean(time2), 2), np.round(np.max(time2), 2)]

    # Confidence Intervals

    interval_acc = 1.96 * math.sqrt((np.round(np.mean(acc) / 100, 4) * (1 - np.round(np.mean(acc) / 100, 4))) / samples)
    interval_prec = 1.96 * math.sqrt(
        (np.round(np.mean(prec) / 100, 4) * (1 - np.round(np.mean(prec) / 100, 4))) / samples)
    interval_rec = 1.96 * math.sqrt((np.round(np.mean(rec) / 100, 4) * (1 - np.round(np.mean(rec) / 100, 4))) / samples)
    interval_f1 = 1.96 * math.sqrt((np.round(np.mean(f1) / 100, 4) * (1 - np.round(np.mean(f1) / 100, 4))) / samples)

    final_acc = [np.round(np.mean(acc), 2), np.round((interval_acc * 100), 2)]
    final_prec = [np.round(np.mean(prec), 2), np.round((interval_prec * 100), 2)]
    final_rec = [np.round(np.mean(rec), 2), np.round((interval_rec * 100), 2)]
    final_f1 = [np.round(np.mean(f1), 2), np.round((interval_f1 * 100), 2)]
    final_time = np.round(np.mean(time2), 2)

    print('\n')
    print('Dataset: ' + str(dataset_title))
    print('Model: ' + str(model_title))
    print('Input features: ' + str(feature_title))
    print('Time: ' + str(final_time))
    print('Accuracy: ' + str(final_acc[0]) + ' +/- ' + str(final_acc[1]))
    print('Precision: ' + str(final_prec[0]) + ' +/- ' + str(final_prec[1]))
    print('Recall: ' + str(final_rec[0]) + ' +/- ' + str(final_rec[1]))
    print('F1 score: ' + str(final_f1[0]) + ' +/- ' + str(final_f1[1]))

    return model_title, final_time, final_acc, final_prec, final_rec, final_f1


if __name__ == '__main__':

    choice = 1

    while choice == 1:

        clear_local_memory()
        clear_global_memory()

        model_title, final_time, final_acc, final_prec, final_rec, final_f1 = main()

        while choice != 1 and choice != 0:
            print('\n')
            choice = int(input('Press 1 to continue or 0 to exit: '))

        if choice == 0:
            break

    print('Done')
