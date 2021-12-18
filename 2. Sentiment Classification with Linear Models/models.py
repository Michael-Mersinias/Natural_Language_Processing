# models.py

from sentiment_data import *
from utils import *

# Importing libraries included in standard Python.
from collections import Counter, OrderedDict
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Importing libraries included in standard Python.
import re
import random

nltk.download('stopwords')
nltk.download('wordnet')

cachedStopWords = stopwords.words("english")
cachedLemmatizer = WordNetLemmatizer()
cachedStemmer = PorterStemmer()


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    # Unigrams - [This, is, an example]
    # Text preprocessing, counter creation and addition to indexer handling
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        preprocess = re.compile('.*[A-Za-z0-9?!.].*')
        filtered = [w.lower() for w in sentence if preprocess.match(w) and w.lower() not in cachedStopWords]
        counter = Counter(filtered)
        for word in counter:
            if not self.get_indexer().contains(word):
                self.get_indexer().add_and_get_index(word, add=add_to_indexer)

        return counter


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    # Bigrams - [This|is, is|an, an|example]
    # Text preprocessing, counter creation and addition to indexer handling
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        preprocess = re.compile('.*[A-Za-z0-9?!.].*')
        filtered = [w.lower() for w in sentence if preprocess.match(w) and w.lower() not in cachedStopWords]

        if not add_to_indexer:
            bigrams = [filtered[i] + '|' + filtered[i + 1] for i in range(len(filtered) - 2 + 1)]
        else:
            bigrams = filtered  # Bigrams have already been created

        counter = Counter(bigrams)
        for word in counter:
            if not self.get_indexer().contains(word):
                self.get_indexer().add_and_get_index(word, add=add_to_indexer)

        return counter


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    # Various methods attempted such as lemmatization, manipulation of the stopword list, count of presence, ngrams, sentence length normalization.
    # Text preprocessing, counter creation and addition to indexer handling
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        customStopWords = set(cachedStopWords)

        customStopWords.difference_update(
            ["not", "no", 'nor', "wasn't", "isn't", "aren't", "wasn't", 'against', "mustn't", "shouldn't", "wouldn't"])
        customStopWords.intersection_update(['the'])

        preprocess = re.compile(".*[A-Za-z0-9?!.'].*")
        filtered = [(cachedLemmatizer.lemmatize(w)).lower() for w in sentence if
                    (preprocess.match(w) and w not in customStopWords)]

        # Method 0 - Better preprocessing -> Lemmatization and manipulation of the stopword set
        # Best accuracy: 77.9%

        # counter = Counter(filtered)

        # Method 1 - Trigrams
        # Best accuracy: 61%

        # if not add_to_indexer:
        #    trigrams = [filtered[i] + '|' + filtered[i + 1] + '|' + filtered[i + 2] for i in range(len(filtered) - 3 + 1)]
        # else:
        #    trigrams = filtered  # Trigrams have already been created

        # counter = Counter(trigrams)

        # Method 3 - Fourgrams
        # Best accuracy: 54%

        # if not add_to_indexer:
        #    fourgrams = [filtered[i] + '|' + filtered[i + 1] + '|' + filtered[i + 2] + '|' + filtered[i + 3] for i in range(len(filtered) - 4 + 1)]
        # else:
        #    fourgrams = filtered  # Trigrams have already been created

        #counter = Counter(fourgrams)

        # Method 2 - Calculate presence instead of counts of each word in each SentimentExample
        # Best accuracy: 79.7%

        counter = Counter(list(OrderedDict.fromkeys(filtered)))

        # Method 3 - Divide the counts of each word by the length (num of words) of each SentimentExample
        # Best accuracy: 72.2%

        # counter = Counter(filtered)
        # length = 1.0 * len(counter)
        # for k in counter:
        #    counter[k] = round(counter[k] / length, 5)

        for word in counter:
            if not self.get_indexer().contains(word):
                self.get_indexer().add_and_get_index(word, add=add_to_indexer)

        return counter


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """

    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """

    def __init__(self, feature_extractor, weights=np.zeros(10000), alpha=0.3):
        self.feature_extractor = feature_extractor
        self.weights = weights
        self.alpha = alpha
        self.features = []
        self.dot_product = 0

    # Make a prediction by calculating w*f(x) for each word that exists (index_of(word)!=-1) in the indexer
    # Return 1 in case w*f(x)>0 and 0 otherwise
    def predict(self, x) -> int:

        self.features = self.feature_extractor.extract_features(x, False)

        dot_product = 0
        for word in self.features:
            if self.feature_extractor.get_indexer().index_of(word) != -1:
                dot_product += self.weights[self.feature_extractor.get_indexer().index_of(word)] * self.features[word]
        self.dot_product = dot_product
        if dot_product > 0:
            return 1
        else:
            return 0

    # Perceptron - Update weights as in the lectures according to the Class Label (y_true) value
    # Call ONLY in case of a wrong (y_pred != y_true) prediction
    def update(self, y_true):
        for word in self.features:
            if self.feature_extractor.get_indexer().index_of(word) != -1:
                if y_true == 1:
                    self.weights[self.feature_extractor.get_indexer().index_of(word)] += self.alpha * self.features[
                        word]
                else:
                    self.weights[self.feature_extractor.get_indexer().index_of(word)] -= self.alpha * self.features[
                        word]


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """

    def __init__(self, feature_extractor, weights=np.zeros(10000), alpha=0.1):
        self.feature_extractor = feature_extractor
        self.weights = weights
        self.alpha = alpha
        self.features = []
        self.dot_product = 0

    # Make a prediction by calculating w*f(x) for each word that exists (index_of(word)!=-1) in the indexer
    # Return 1 in case w*f(x)>0, equivalent to P(y=+1|x)>0.5) and 0 otherwise
    def predict(self, x) -> int:

        self.features = self.feature_extractor.extract_features(x, False)

        dot_product = 0
        for word in self.features:
            if self.feature_extractor.get_indexer().index_of(word) != -1:
                dot_product += self.weights[self.feature_extractor.get_indexer().index_of(word)] * self.features[word]
        self.dot_product = dot_product
        if dot_product > 0:
            return 1
        else:
            return 0

    # Logistic Regression - Update weights as in the lectures according to the Class Label (y_true) value
    # Call in both cases of a correct (y_pred == y_true) and a wrong (y_pred != y_true) prediction
    def update(self, y_true):
        for word in self.features:
            if self.feature_extractor.get_indexer().index_of(word) != -1:
                if y_true == 1:
                    self.weights[self.feature_extractor.get_indexer().index_of(word)] += self.alpha * self.features[
                        word] * (1 - (np.exp(self.dot_product) / (1 + np.exp(self.dot_product))))
                else:
                    self.weights[self.feature_extractor.get_indexer().index_of(word)] -= self.alpha * self.features[
                        word] * (1 - (1 / (1 + np.exp(self.dot_product))))


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """

    # Iterate through all SentimentExample elements of the training set
    # Extract features and sum the resulted Counter objects from each SentimentExample element
    # Convert to a list which will contain all the words that will be added to the indexer
    total_counter = Counter()
    for sample in train_exs:
        counter = feat_extractor.extract_features(sample.words, False)
        total_counter.update(counter)

    # total_counter = total_counter.most_common(10000)
    total_list = list(dict(total_counter).keys())
    feat_extractor.extract_features(total_list, True)

    # Perceptron
    a = 0.15
    model = PerceptronClassifier(feat_extractor, alpha=a, weights=np.zeros(len(total_list)))

    # For each epoch, shuffle the data, adjust learning rate, make a prediction and update weights
    # item.words - The SentimentExample sample (Input)
    # item.label - The Class Label of the SentimentExample sample
    # y_pred - The Class Label prediction of the Perceptron model
    # In the end, calculate the average accuracy
    epochs = 20
    for epoch in range(epochs):
        print("Epoch: %d" % epoch)
        acc = []
        random.shuffle(train_exs)
        model.alpha = a / (epoch + 1)
        for item in train_exs:
            y_pred = model.predict(item.words)
            if y_pred != item.label:
                model.update(item.label)
                acc.append(0)
            else:
                acc.append(1)
        print("Epoch %d - Accuracy: %f" % (epoch, (sum(acc) / len(acc))))

    return model


def train_logistic_regression(train_exs: List[SentimentExample],
                              feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """

    # Iterate through all SentimentExample elements of the training set
    # Extract features and sum the resulted Counter objects from each SentimentExample element
    # Convert to a list which will contain all the words that will be added to the indexer
    total_counter = Counter()
    for sample in train_exs:
        counter = feat_extractor.extract_features(sample.words, False)
        total_counter.update(counter)

    total_counter = total_counter.most_common(10000)
    total_list = list(dict(total_counter).keys())
    feat_extractor.extract_features(total_list, True)

    # Logistic Regression
    a = 0.3
    model = LogisticRegressionClassifier(feat_extractor, alpha=a, weights=np.zeros(len(total_list)))

    # For each epoch, shuffle the data, adjust learning rate, make a prediction and update weights
    # item.words - The SentimentExample sample (Input)
    # item.label - The Class Label of the SentimentExample sample
    # y_pred - The Class Label prediction of the Logistic Regression model
    # In the end, calculate the average accuracy
    epochs = 20
    for epoch in range(epochs):
        print("Epoch: %d" % epoch)
        acc = []
        random.shuffle(train_exs)
        model.alpha = a / (epoch + 1)
        for item in train_exs:
            y_pred = model.predict(item.words)
            model.update(item.label)
            if y_pred != item.label:
                acc.append(0)
            else:
                acc.append(1)
        print("Epoch %d - Accuracy: %f" % (epoch, (sum(acc) / len(acc))))

    return model


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model
