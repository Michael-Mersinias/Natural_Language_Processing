# models.py

from optimizers import *
from nerdata import *
from utils import *

import random
import time

from collections import Counter
from typing import List

import numpy as np


class ProbabilisticSequenceScorer(object):
    """
    Scoring function for sequence models based on conditional probabilities.
    Scores are provided for three potentials in the model: initial scores (applied to the first tag),
    emissions, and transitions. Note that CRFs typically don't use potentials of the first type.

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """
    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs: np.ndarray, transition_log_probs: np.ndarray, emission_log_probs: np.ndarray):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def score_init(self, sentence_tokens: List[Token], tag_idx: int):
        return self.init_log_probs[tag_idx]

    def score_transition(self, sentence_tokens: List[Token], prev_tag_idx: int, curr_tag_idx: int):
        return self.transition_log_probs[prev_tag_idx, curr_tag_idx]

    def score_emission(self, sentence_tokens: List[Token], tag_idx: int, word_posn: int):
        word = sentence_tokens[word_posn].word
        word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.index_of("UNK")
        return self.emission_log_probs[tag_idx, word_idx]


class HmmNerModel(object):
    """
    HMM NER model for predicting tags

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """
    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs, transition_log_probs, emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def decode(self, sentence_tokens: List[Token]) -> LabeledSentence:
        """
        See BadNerModel for an example implementation
        :param sentence_tokens: List of the tokens in the sentence to tag
        :return: The LabeledSentence consisting of predictions over the sentence
        """

        # Declaring the ProbabilisticSequenceScorer and calling the custom viterbi function to conduct Viterbi based decoding

        scorer = ProbabilisticSequenceScorer(self.tag_indexer, self.word_indexer, self.init_log_probs, self.transition_log_probs, self.emission_log_probs)

        return viterbi(sentence_tokens, self.tag_indexer, scorer)


def train_hmm_model(sentences: List[LabeledSentence], silent: bool=False) -> HmmNerModel:
    """
    Uses maximum-likelihood estimation to read an HMM off of a corpus of sentences.
    Any word that only appears once in the corpus is replaced with UNK. A small amount
    of additive smoothing is applied.
    :param sentences: training corpus of LabeledSentence objects
    :return: trained HmmNerModel
    """
    # Index words and tags. We do this in advance so we know how big our
    # matrices need to be.
    tag_indexer = Indexer()
    word_indexer = Indexer()
    word_indexer.add_and_get_index("UNK")
    word_counter = Counter()
    for sentence in sentences:
        for token in sentence.tokens:
            word_counter[token.word] += 1.0
    for sentence in sentences:
        for token in sentence.tokens:
            # If the word occurs fewer than two times, don't index it -- we'll treat it as UNK
            get_word_index(word_indexer, word_counter, token.word)
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)
    # Count occurrences of initial tags, transitions, and emissions
    # Apply additive smoothing to avoid log(0) / infinities / etc.
    init_counts = np.ones((len(tag_indexer)), dtype=float) * 0.001
    transition_counts = np.ones((len(tag_indexer),len(tag_indexer)), dtype=float) * 0.001
    emission_counts = np.ones((len(tag_indexer),len(word_indexer)), dtype=float) * 0.001
    for sentence in sentences:
        bio_tags = sentence.get_bio_tags()
        for i in range(0, len(sentence)):
            tag_idx = tag_indexer.add_and_get_index(bio_tags[i])
            word_idx = get_word_index(word_indexer, word_counter, sentence.tokens[i].word)
            emission_counts[tag_idx][word_idx] += 1.0
            if i == 0:
                init_counts[tag_idx] += 1.0
            else:
                transition_counts[tag_indexer.add_and_get_index(bio_tags[i-1])][tag_idx] += 1.0
    # Turn counts into probabilities for initial tags, transitions, and emissions. All
    # probabilities are stored as log probabilities
    if not silent:
        print(repr(init_counts))
    init_counts = np.log(init_counts / init_counts.sum())
    # transitions are stored as count[prev state][next state], so we sum over the second axis
    # and normalize by that to get the right conditional probabilities
    transition_counts = np.log(transition_counts / transition_counts.sum(axis=1)[:, np.newaxis])
    # similar to transitions
    emission_counts = np.log(emission_counts / emission_counts.sum(axis=1)[:, np.newaxis])
    if not silent:
        print("Tag indexer: %s" % tag_indexer)
        print("Initial state log probabilities: %s" % init_counts)
        print("Transition log probabilities: %s" % transition_counts)
        print("Emission log probs too big to print...")
        print("Emission log probs for India: %s" % emission_counts[:,word_indexer.add_and_get_index("India")])
        print("Emission log probs for Phil: %s" % emission_counts[:,word_indexer.add_and_get_index("Phil")])
        print("   note that these distributions don't normalize because it's p(word|tag) that normalizes, not p(tag|word)")
    return HmmNerModel(tag_indexer, word_indexer, init_counts, transition_counts, emission_counts)


def get_word_index(word_indexer: Indexer, word_counter: Counter, word: str) -> int:
    """
    Retrieves a word's index based on its count. If the word occurs only once, treat it as an "UNK" token
    At test time, unknown words will be replaced by UNKs.
    :param word_indexer: Indexer mapping words to indices for HMM featurization
    :param word_counter: Counter containing word counts of training set
    :param word: string word
    :return: int of the word index
    """
    if word_counter[word] < 1.5:
        return word_indexer.add_and_get_index("UNK")
    else:
        return word_indexer.add_and_get_index(word)


##################
# CRF code follows

class FeatureBasedSequenceScorer(object):
    """
    Feature-based sequence scoring model. Note that this scorer is instantiated *for every example*: it contains
    the feature cache used for that example.
    """
    def __init__(self, tag_indexer, feature_weights, feat_cache):
        self.tag_indexer = tag_indexer
        self.feature_weights = feature_weights
        self.feat_cache = feat_cache

    def score_init(self, sentence, tag_idx):
        if isI(self.tag_indexer.get_object(tag_idx)):
            return -1000
        else:
            return 0

    def score_transition(self, sentence_tokens, prev_tag_idx, curr_tag_idx):
        prev_tag = self.tag_indexer.get_object(prev_tag_idx)
        curr_tag = self.tag_indexer.get_object(curr_tag_idx)
        if (isO(prev_tag) and isI(curr_tag))\
                or (isB(prev_tag) and isI(curr_tag) and get_tag_label(prev_tag) != get_tag_label(curr_tag)) \
                or (isI(prev_tag) and isI(curr_tag) and get_tag_label(prev_tag) != get_tag_label(curr_tag)):
            return -1000
        else:
            return 0

    def score_emission(self, sentence_tokens, tag_idx, word_posn):
        feats = self.feat_cache[word_posn][tag_idx]
        return self.feature_weights.score(feats)


class CrfNerModel(object):
    def __init__(self, tag_indexer, feature_indexer, feature_weights):
        self.tag_indexer = tag_indexer
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights

    def decode(self, sentence_tokens: List[Token]) -> LabeledSentence:
        """
        See BadNerModel for an example implementation
        :param sentence_tokens: List of the tokens in the sentence to tag
        :return: The LabeledSentence consisting of predictions over the sentence
        """

        # Creating the feature cache and the FeatureBasedSequenceScorer as in the train_crf_model function
        # Then we the custom viterbi function to conduct Viterbi based decoding

        feat_cache = [[[] for j in range(len(self.tag_indexer))] for i in range(len(sentence_tokens))]

        for i in range(len(sentence_tokens)):
            for j in range(len(self.tag_indexer)):
                feat_cache[i][j] = extract_emission_features(sentence_tokens, i, self.tag_indexer.get_object(j), self.feature_indexer, False)

        scorer = FeatureBasedSequenceScorer(self.tag_indexer, self.feature_weights, feat_cache)

        return viterbi(sentence_tokens, self.tag_indexer, scorer)

    def decode_beam(self, sentence_tokens: List[Token]) -> LabeledSentence:
        """
        See BadNerModel for an example implementation
        :param sentence_tokens: List of the tokens in the sentence to tag
        :return: The LabeledSentence consisting of predictions over the sentence
        """

        # Creating the feature cache and the FeatureBasedSequenceScorer as in the train_crf_model function
        # Then we the custom beam_search function to conduct Beam Search based decoding

        feat_cache = [[[] for j in range(len(self.tag_indexer))] for i in range(len(sentence_tokens))]

        for i in range(len(sentence_tokens)):
            for j in range(len(self.tag_indexer)):
                feat_cache[i][j] = extract_emission_features(sentence_tokens, i, self.tag_indexer.get_object(j), self.feature_indexer, False)

        scorer = FeatureBasedSequenceScorer(self.tag_indexer, self.feature_weights, feat_cache)

        return beam_search(sentence_tokens, self.tag_indexer, scorer)


def train_crf_model(sentences: List[LabeledSentence], silent: bool=False) -> CrfNerModel:
    """
    Trains a CRF NER model on the given corpus of sentences.
    :param sentences: The training data
    :param silent: True to suppress output, false to print certain debugging outputs
    :return: The CrfNerModel, which is primarily a wrapper around the tag + feature indexers as well as weights
    """
    tag_indexer = Indexer()
    for sentence in sentences:
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)
    if not silent:
        print("Extracting features")
    feature_indexer = Indexer()
    # 4-d list indexed by sentence index, word index, tag index, feature index
    feature_cache = [[[[] for k in range(0, len(tag_indexer))] for j in range(0, len(sentences[i]))] for i in range(0, len(sentences))]
    for sentence_idx in range(0, len(sentences)):
        if sentence_idx % 100 == 0 and not silent:
            print("Ex %i/%i" % (sentence_idx, len(sentences)))
        for word_idx in range(0, len(sentences[sentence_idx])):
            for tag_idx in range(0, len(tag_indexer)):
                feature_cache[sentence_idx][word_idx][tag_idx] = extract_emission_features(sentences[sentence_idx].tokens, word_idx, tag_indexer.get_object(tag_idx), feature_indexer, add_to_indexer=True)
    if not silent:
        print("Training")
    weight_vector = UnregularizedAdagradTrainer(np.zeros((len(feature_indexer))), eta=1.0)
    num_epochs = 3
    random.seed(0)
    for epoch in range(0, num_epochs):
        epoch_start = time.time()
        if not silent:
            print("Epoch %i" % epoch)
        sent_indices = [i for i in range(0, len(sentences))]
        random.shuffle(sent_indices)
        total_obj = 0.0
        for counter, i in enumerate(sent_indices):
            if counter % 100 == 0 and not silent:
                print("Ex %i/%i" % (counter, len(sentences)))
            scorer = FeatureBasedSequenceScorer(tag_indexer, weight_vector, feature_cache[i])
            (gold_log_prob, gradient) = compute_gradient(sentences[i], tag_indexer, scorer, feature_indexer)
            total_obj += gold_log_prob
            weight_vector.apply_gradient_update(gradient, 1)
        if not silent:
            print("Objective for epoch: %.2f in time %.2f" % (total_obj, time.time() - epoch_start))
    return CrfNerModel(tag_indexer, feature_indexer, weight_vector)


def extract_emission_features(sentence_tokens: List[Token], word_index: int, tag: str, feature_indexer: Indexer, add_to_indexer: bool):
    """
    Extracts emission features for tagging the word at word_index with tag.
    :param sentence_tokens: sentence to extract over
    :param word_index: word index to consider
    :param tag: the tag that we're featurizing for
    :param feature_indexer: Indexer over features
    :param add_to_indexer: boolean variable indicating whether we should be expanding the indexer or not. This should
    be True at train time (since we want to learn weights for all features) and False at test time (to avoid creating
    any features we don't have weights for).
    :return: an ndarray
    """
    feats = []
    curr_word = sentence_tokens[word_index].word
    # Lexical and POS features on this word, the previous, and the next (Word-1, Word0, Word1)
    for idx_offset in range(-1, 2):
        if word_index + idx_offset < 0:
            active_word = "<s>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_word = "</s>"
        else:
            active_word = sentence_tokens[word_index + idx_offset].word
        if word_index + idx_offset < 0:
            active_pos = "<S>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_pos = "</S>"
        else:
            active_pos = sentence_tokens[word_index + idx_offset].pos
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Word" + repr(idx_offset) + "=" + active_word)
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Pos" + repr(idx_offset) + "=" + active_pos)
    # Character n-grams of the current word
    max_ngram_size = 3
    for ngram_size in range(1, max_ngram_size+1):
        start_ngram = curr_word[0:min(ngram_size, len(curr_word))]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":StartNgram=" + start_ngram)
        end_ngram = curr_word[max(0, len(curr_word) - ngram_size):]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":EndNgram=" + end_ngram)
    # Look at a few word shape features
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":IsCap=" + repr(curr_word[0].isupper()))
    # Compute word shape
    new_word = []
    for i in range(0, len(curr_word)):
        if curr_word[i].isupper():
            new_word += "X"
        elif curr_word[i].islower():
            new_word += "x"
        elif curr_word[i].isdigit():
            new_word += "0"
        else:
            new_word += "?"
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape=" + repr(new_word))
    return np.asarray(feats, dtype=int)


def compute_gradient(sentence: LabeledSentence, tag_indexer: Indexer, scorer: FeatureBasedSequenceScorer, feature_indexer: Indexer) -> (float, Counter):
    """
    Computes the gradient of the given example (sentence). The bulk of this code will be computing marginals via
    forward-backward: you should first compute these marginals, then accumulate the gradient based on the log
    probabilities.
    :param sentence: The LabeledSentence of the current example
    :param tag_indexer: The Indexer of the tags
    :param scorer: FeatureBasedSequenceScorer is a scoring model that wraps the weight vector and which also contains a
    feat_cache field that will be useful when computing the gradient.
    :param feature_indexer: The Indexer of the features
    :return: A tuple of two items. The first is the log probability of the correct sequence, which corresponds to the
    training objective. This value is only needed for printing, so technically you do not *need* to return it, but it
    will probably be useful to compute for debugging purposes.
    The second value is a Counter containing the gradient -- this is a sparse map from indices (features)
    to weights (gradient values).
    """

    # Number of tokens, e.g. 4 - [Token(Results, NNS, I-NP), Token(of, IN, I-PP), Token(South, JJ, I-NP), Token(Korean, JJ, I-NP)]
    tokens = len(sentence.tokens)
    # Number of tags (always 9) - ['O', 'B-ORG', 'B-MISC', 'B-PER', 'I-PER', 'B-LOC', 'I-ORG', 'I-MISC', 'I-LOC']
    tags = len(tag_indexer)

    # Forward matrix and Backward matrix - For both: Rows = #tags (always 9), Columns = #tokens (words)
    # Initialize to zeroes
    forward = np.zeros(shape=(tags, tokens))
    backward = np.zeros(shape=(tags, tokens))

    # Caching
    init_matrix = np.zeros(tags)
    transition_matrix = np.zeros(shape=(tags, tags))
    emission_matrix = np.zeros(shape=(tags, tokens))

    for i in range(tokens):
        for j in range(tags):
            init_matrix[j] = scorer.score_init(sentence.tokens, j)
            emission_matrix[j, i] = scorer.score_emission(sentence.tokens, j, i)
            for k in range(tags):
                transition_matrix[k, j] = scorer.score_transition(sentence.tokens, k, j)


    # Forward -> Similar to the Viterbi algorithm but we sum the elements instead of taking the max

    for j in range(tags):
        forward[j, 0] = init_matrix[j] + emission_matrix[j, 0]

    for i in range(1, tokens):
        for j in range(tags):
            sum_value = forward[0, i - 1] + transition_matrix[0, j] + emission_matrix[j, i]
            for k in range(1, tags):
                sum_value = np.logaddexp(sum_value, (forward[k, i-1] + transition_matrix[k, j] + emission_matrix[j, i]))
            forward[j, i] = sum_value

    # Backward -> Analogous to Forward but we count emission for the next (i+1) timestep instead

    for i in range(tokens-2, -1, -1):
        for j in range(tags):
            sum_value = backward[0, i + 1] + transition_matrix[j, 0] + emission_matrix[0, i+1]
            for k in range(1, tags):
                sum_value = np.logaddexp(sum_value, (backward[k, i + 1] + transition_matrix[j, k] + emission_matrix[k, i+1]))
            backward[j, i] = sum_value

    # Computing Marginals as in the lecture
    # We perform the calculations in the log space by using np.logandexp, as (+, x) in real space translates to (log-sum-exp, +) in log space (Example: e^2 * e^3 = e^(2+3) = e^5)
    # The nominator is the logaddexp sum of the forward alpha and the backward beta of the respective element
    # The denominator for normalization is the logaddexp sum of the forward alpha and the backward beta of all elements

    denominator = np.zeros(tokens)
    for i in range(tokens):
        sum_value = forward[0, i] + backward[0, i]
        for j in range(1, tags):
            sum_value = np.logaddexp(sum_value, (forward[j][i] + backward[j][i]))
        denominator[i] = sum_value

    marginals = np.zeros(shape=(tags, tokens))
    for i in range(tokens):
        for j in range(tags):
            marginals[j, i] = np.exp(forward[j, i] + backward[j, i] - denominator[i])

    # Computing the accumulating gradient
    # For emission features only: we simply add the gold features and subtract the marginals as in the lecture
    # Thus, Gradient = Gold Features - Marginals (expected features under model)

    total_score = 0
    gradient = Counter()
    for i in range(tokens):

        gold_index = tag_indexer.index_of(sentence.get_bio_tags()[i])
        total_score += emission_matrix[gold_index, i]

        features_cache = scorer.feat_cache[i][gold_index]
        for k in features_cache:
            gradient[k] += 1

        for j in range(tags):
            features_cache = scorer.feat_cache[i][j]
            for k in features_cache:
                gradient[k] -= marginals[j, i]

    return total_score, gradient


def viterbi(sentence_tokens, tag_indexer, scorer) -> LabeledSentence:

    # Number of tokens, e.g. 4 - [Token(Results, NNS, I-NP), Token(of, IN, I-PP), Token(South, JJ, I-NP), Token(Korean, JJ, I-NP)]
    tokens = len(sentence_tokens)
    # Number of tags (always 9) - ['O', 'B-ORG', 'B-MISC', 'B-PER', 'I-PER', 'B-LOC', 'I-ORG', 'I-MISC', 'I-LOC']
    tags = len(tag_indexer)

    # Viterbi matrix and Backpointer matrix - For both: Rows = #tags (always 9), Columns = #tokens (words)
    # Initialize to zeroes
    viterbi_matrix = np.zeros(shape=(tags, tokens))
    backpointer = np.zeros(shape=(tags, tokens), dtype=int)

    # For token=0, we have no transition, so we use score_init instead to compute the score as initial score + emission score

    for j in range(tags):
        viterbi_matrix[j, 0] = scorer.score_init(sentence_tokens, j) + scorer.score_emission(sentence_tokens, j, 0)

    # For the rest of the tokens, we compute all transition combinations and the respective scores (previous score + transition score + emission score)
    # We keep the maximum score in the Viterbi matrix and we also store the index of the maximum score in the Backpointer matrix

    for i in range(1, tokens):
        for j in range(tags):
            tag_list = []
            candidates = []
            emission = scorer.score_emission(sentence_tokens, j, i)
            for k in range(tags):
                candidates.append(viterbi_matrix[k, i - 1] + scorer.score_transition(sentence_tokens, k, j) + emission)
                tag_list.append(k)
            viterbi_matrix[j, i] = np.max(candidates)
            backpointer[j, i] = tag_list[np.argmax(candidates)]

    # Calculating the most likely (the one which provides the max score) state sequence
    # Starting at the max score value (max value of the last Viterbi matrix column), we find the optimal path going backwards
    # We make use of the Backpointer matrix which has stored the indices of path leading to the max score

    pred_tags = []
    backpointed_index = np.argmax(viterbi_matrix[:, tokens - 1])
    pred_tags.append(tag_indexer.get_object(backpointed_index))
    for i in range(tokens-1, 0, -1):
        backpointed_index = backpointer[backpointed_index, i]
        pred_tags.append(tag_indexer.get_object(backpointed_index))

    pred_tags.reverse()
    chunks = chunks_from_bio_tag_seq(pred_tags)

    return LabeledSentence(sentence_tokens, chunks)


def beam_search(sentence_tokens, tag_indexer, scorer) -> LabeledSentence:

    # Number of tokens, e.g. 4 - [Token(Results, NNS, I-NP), Token(of, IN, I-PP), Token(South, JJ, I-NP), Token(Korean, JJ, I-NP)]
    tokens = len(sentence_tokens)
    # Number of tags (always 9) - ['O', 'B-ORG', 'B-MISC', 'B-PER', 'I-PER', 'B-LOC', 'I-ORG', 'I-MISC', 'I-LOC']
    tags = len(tag_indexer)

    # Backpointer matrix : Rows = #tags (always 9), Columns = #tokens (words)
    # Initialize to zeroes
    backpointer = np.zeros(shape=(tags, tokens), dtype=int)

    beam_list = []
    beam_size = 2

    # For token=0, we have no transition, so we use score_init instead to compute the score as initial score + emission score

    beam = Beam(beam_size)
    for j in range(tags):
        beam.add(j, scorer.score_init(sentence_tokens, j) + scorer.score_emission(sentence_tokens, j, 0))
    beam_list.append(beam)

    # For the rest of the tokens, we compute the transition combinations but we only take into account the top 2 previous states
    # The respective scores (previous score + transition score + emission score) are calculated afterwards
    # We keep the maximum score and its respective tag index in the Beam and we also store the index of the maximum score in the Backpointer matrix
    # We finally append the Beam to a beam list which has a length equal to the numbers of sentence tokens

    for i in range(1, tokens):
        beam = Beam(beam_size)
        for j in range(tags):
            candidates = []
            tags_list = []
            for tag, score in beam_list[i-1].get_elts_and_scores():
                candidates.append(score + scorer.score_transition(sentence_tokens, tag, j))
                tags_list.append(tag)
            beam.add(j, (np.max(candidates) + scorer.score_emission(sentence_tokens, j, i)))
            backpointer[j, i] = tags_list[np.argmax(candidates)]
        beam_list.append(beam)

    # Calculating the most likely (the one which provides the max score) state sequence
    # Starting at the max score value (first element of the last beam added to the beam_list), we find the optimal path going backwards through the beam list
    # In most cases, the optimal backward path is through the previous beam's head (element at index 0), but not always
    # To tackle this, we make use of the Backpointer matrix which has stored the indices of path leading to the max score
    # We make a check by using the Backpointer matrix and if required, we write the correct value at the previous beam's head (index 0)

    pred_tags = []

    for i in range(tokens-1, 0, -1):
        if backpointer[beam_list[i].head(), i] != beam_list[i - 1].head():
            beam_list[i - 1].elts[0] = backpointer[beam_list[i].head(), i]
        pred_tags.append(tag_indexer.get_object(beam_list[i].head()))
    pred_tags.append(tag_indexer.get_object(beam_list[0].head()))

    pred_tags.reverse()
    chunks = chunks_from_bio_tag_seq(pred_tags)

    return LabeledSentence(sentence_tokens, chunks)
