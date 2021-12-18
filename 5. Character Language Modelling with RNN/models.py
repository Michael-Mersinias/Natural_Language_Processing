# models.py

import numpy as np
import collections
import torch
import random

#####################
# MODELS FOR PART 1 #
#####################
from utils import Indexer


class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :param context:
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Only implemented in subclasses")


class FrequencyBasedClassifier(ConsonantVowelClassifier):
    """
    Classifier based on the last letter before the space. If it has occurred with more consonants than vowels,
    classify as consonant, otherwise as vowel.
    """
    def __init__(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context):
        # Look two back to find the letter before the space
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1


class RNNClassifier(ConsonantVowelClassifier):

    def __init__(self, indexer, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        ConsonantVowelClassifier.__init__(self)
        self.indexer = indexer
        self.model = RNN(len(indexer), input_size, hidden_size, num_layers, output_size)

    def predict(self, context):
        char_indices = []
        for char in context:
            if self.indexer.index_of(char) > -1:
                char_indices.append(self.indexer.index_of(char))
            else:
                char_indices.append(-1)
        chars_indices_tensor = torch.tensor([char_indices])
        probs = self.model.forward(chars_indices_tensor)
        return torch.argmax(probs)


def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)


# Support function: takes a batch containing a list of sentences and maps all characters to indices through the Indexer

def mapping(batch_ex_words: list, indexer: Indexer):
    final_list = []
    for i in range(len(batch_ex_words)):
        tmp = []
        for char in batch_ex_words[i]:
            if indexer.index_of(char) > -1:
                tmp.append(indexer.index_of(char))
            else:
                tmp.append(-1)
        final_list.append(tmp)
    return final_list


def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_cons_exs: list of strings followed by consonants
    :param train_vowel_exs: list of strings followed by vowels
    :param dev_cons_exs: list of strings followed by consonants
    :param dev_vowel_exs: list of strings followed by vowels
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNClassifier instance trained on the given data
    """

    # Declare parameters

    input_size = 64
    hidden_size = 32
    num_layers = 1
    output_size = 2

    rnn = RNNClassifier(vocab_index, input_size, hidden_size, num_layers, output_size)

    num_epochs = 30
    batch_size = 128

    lr = 0.005
    optimizer = torch.optim.Adam(rnn.model.parameters(), lr=lr)

    # Merge the two train_exs lists (strings followed by consonants and strings followed by vowels)
    # Then create an analogous list with their corresponding labels (0 and 1 for string followed by consonants and vowels respectively)
    train_exs = train_cons_exs + train_vowel_exs
    labels_y = [0] * len(train_cons_exs) + [1] * len(train_vowel_exs)

    # Generate an index for each sentence of the merged train_exs list.
    dataset_ex_indices = [i for i in range(0, len(train_exs))]

    for epoch in range(0, num_epochs):
        print("Epoch: %d" % epoch)
        # New Epoch - Shuffle the data and adjust learning rate accordingly
        if (epoch+1) % 10 == 0:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / (epoch + 1)
        random.shuffle(dataset_ex_indices)
        total_loss = 0.0

        for i in range(0, len(dataset_ex_indices), batch_size):
            # Select a batch of indices from the dataset, of size equal to the batch_size parameter
            batch_ex_indices = dataset_ex_indices[i: i + batch_size]

            # Creating two lists for the batch: a list of sentences and a list of labels
            batch_ex_chars = []
            batch_ex_labels = []
            for j in range(len(batch_ex_indices)):
                batch_ex_chars.append(train_exs[batch_ex_indices[j]])
                batch_ex_labels.append(labels_y[batch_ex_indices[j]])

            # Map characters to indices through the Indexer by making use of the "mapping" function
            # Thus transforming the list of sentences to a list of lists, each element corresponding to a character-mapped integer value
            batch_ex_char_indexer_values = mapping(batch_ex_chars, rnn.indexer)

            # Print an example and make sure all of the lists are of size equal to the batch_size parameter
            # print(batch_ex_indices[0], batch_ex_chars[0], batch_ex_char_indexer_values[0], batch_ex_labels[0])
            # print(len(batch_ex_indices), len(batch_ex_chars), len(batch_ex_char_indexer_values), len(batch_ex_labels))

            # Transform the lists to tensors
            tensor_x = torch.tensor(batch_ex_char_indexer_values)
            tensor_y = torch.tensor(batch_ex_labels)

            # Train the model as per the lectures and calculate the total loss for the corresponding epoch
            rnn.model.train()
            rnn.model.zero_grad()
            probs = rnn.model.forward(tensor_x)
            loss = rnn.model.loss(probs, tensor_y)
            total_loss += loss
            loss.backward()
            optimizer.step()
        print("Total loss on epoch %i: %f" % (epoch, total_loss))

    return rnn


# The recurrent neural network architecture as per the lectures
# Layers: Embedding -> GRU -> Feedforward (Linear+Linear) -> Softmax
# Glorot weight initialization for the linear layers
# The selected loss function is CrossEntropyLoss

class RNN(torch.nn.Module):
    def __init__(self, vocab_indexer_size, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.char_embeddings = torch.nn.Embedding(vocab_indexer_size, input_size)
        self.char_embeddings.weight.requires_grad = True

        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.d = torch.nn.Dropout(0.1)
        self.v = torch.nn.Linear(hidden_size, int(hidden_size / 2))
        self.w = torch.nn.Linear(int(hidden_size / 2), output_size)
        self.softmax = torch.nn.LogSoftmax(dim=0)

        self.loss = torch.nn.CrossEntropyLoss()

        torch.nn.init.xavier_uniform_(self.v.weight)
        torch.nn.init.xavier_uniform_(self.w.weight)

    def forward(self, x):
        x = self.char_embeddings(x)  # (batch_size, sequence length, embedding_size)
        output, h_n = self.gru(x)  # (num_layers * directions (1), batch_size, hidden_size)
        h_n = torch.squeeze(h_n)  # (batch_size, hidden_size)
        return self.softmax(self.w((self.v(self.d(h_n)))))

#####################
# MODELS FOR PART 2 #
#####################


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e
        :param context: a single character to score
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class RNNLanguageModel(LanguageModel):
    def __init__(self, indexer, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        LanguageModel.__init__(self)
        self.indexer = indexer
        self.model = RNN2(len(indexer), input_size, hidden_size, num_layers, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=0)

    def get_next_char_log_probs(self, context):
        char_indices = mapping2(context, self.indexer)
        chars_indices_tensor = torch.tensor([char_indices])
        h_n = self.model.forward(chars_indices_tensor)[1]
        return self.softmax(h_n).detach().numpy()

    def get_log_prob_sequence(self, next_chars, context):

        sequence = context + next_chars
        sequence_indices = mapping2(sequence, self.indexer)
        sequence_indices_tensor = torch.tensor([sequence_indices])

        output = self.model.forward(sequence_indices_tensor)[0]
        output = torch.squeeze(output)

        prob_sum = self.softmax(output[len(context)-1, :])[sequence_indices[len(context)]]
        context_index = len(context)
        for next_char_index in sequence_indices[len(context)+1:len(sequence)]:
            prob_sum += self.softmax(output[context_index, :])[next_char_index]
            context_index += 1

        return prob_sum.detach().numpy().item()


# Support function: takes a batch containing a list of characters and maps them to indices through the Indexer

def mapping2(batch_ex_chars: list, indexer: Indexer):
    final_list = []
    for char in batch_ex_chars:
        if indexer.index_of(char) > -1:
            final_list.append(indexer.index_of(char))
        else:
            final_list.append(-1)
    return final_list


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """

    # Declare parameters

    input_size = 128
    hidden_size = 64
    num_layers = 1
    output_size = len(vocab_index)

    rnn = RNNLanguageModel(vocab_index, input_size, hidden_size, num_layers, output_size)

    num_epochs = 30
    batch_size = 50
    chunk_size = 50

    lr = 0.02
    optimizer = torch.optim.Adam(rnn.model.parameters(), lr=lr)

    # Generate an index for each character of the train_text list.
    dataset_ex_indices = [i for i in range(0, len(train_text))]

    for epoch in range(num_epochs):
        print("Epoch: %d" % epoch)
        # New Epoch - Adjust learning rate accordingly
        if (epoch+1) % 10 == 0:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / 2
        total_loss = 0.0
        for i in range(0, len(dataset_ex_indices), batch_size*chunk_size):
            batch_x = []
            batch_y = []

            # Select a number of characters-based indices from the dataset, of size equal to the chunk_size*batch_size parameter
            batch_ex_indices = dataset_ex_indices[i: i + chunk_size * batch_size]

            # Creating a number of chunks equal to the batch_size parameter
            for k in range(i, i + chunk_size * batch_size, chunk_size):

                # Initializing each chunk with a space character on the chars list and element 0 on the labels list
                batch_ex_chars = []
                batch_ex_labels = []
                batch_ex_chars.append(' ')
                batch_ex_labels.append(train_text[batch_ex_indices[0]])

                for j in range(k, k + chunk_size - 1):

                    # Adding characters to the chunk consecutively
                    # For each element i that is added to the chars list, the element i+1 is added to the labels list
                    if j < len(dataset_ex_indices)-1:
                        batch_ex_chars.append(train_text[batch_ex_indices[j-i]])
                        batch_ex_labels.append(train_text[batch_ex_indices[j-i+1]])

                    # Handle cases where there are no more characters to add to the chunk
                    # In this case, we perform padding with space characters
                    else:
                        batch_ex_chars.append(' ')
                        if j == len(dataset_ex_indices)-1:
                            batch_ex_labels.append(train_text[batch_ex_indices[j-i]])
                        else:
                            batch_ex_labels.append(' ')

                # Map characters to indices through the Indexer by making use of the "mapping2" function
                # Thus transforming the list of characters to a list of character-mapped integer values
                # We perform this operation for both the chars and the labels lists
                batch_ex_char_indexer_values = mapping2(batch_ex_chars, rnn.indexer)
                batch_ex_label_indexer_values = mapping2(batch_ex_labels, rnn.indexer)

                # Print an example and make sure all of the lists are of size equal to the batch_size parameter
                # print(batch_ex_chars[0], batch_ex_char_indexer_values[0], batch_ex_labels[0], batch_ex_label_indexer_values[0])
                # print(len(batch_ex_chars), len(batch_ex_char_indexer_values), len(batch_ex_labels), len(batch_ex_char_indexer_values))

                # As long as our batch has enough space for another chunk, add the lists of that chunk (chars and labels) to the corresponding batch lists
                if len(batch_x) < batch_size and len(batch_y) < batch_size:
                    batch_x.append(batch_ex_char_indexer_values)
                    batch_y.append(batch_ex_label_indexer_values)

            # Transform the lists to tensors
            tensor_x = torch.tensor(batch_x)
            tensor_y = torch.tensor(batch_y)

            # Train the model as per the lectures and calculate the total loss for the corresponding epoch
            rnn.model.train()
            rnn.model.zero_grad()
            probs = rnn.model.forward(tensor_x)[0]
            probs = probs.view(batch_size*chunk_size, len(vocab_index))
            tensor_y = tensor_y.view(batch_size*chunk_size)
            loss = rnn.model.loss(probs, tensor_y)
            total_loss += loss
            loss.backward()
            optimizer.step()
        print("Total loss on epoch %i: %f" % (epoch, total_loss))
    return rnn


# The recurrent neural network architecture as per the lectures
# Layers: Embedding -> GRU -> Feedforward (Linear+Linear)
# Glorot weight initialization for the linear layers
# The selected loss function is CrossEntropyLoss

class RNN2(torch.nn.Module):
    def __init__(self, vocab_indexer_size, input_size, hidden_size, num_layers, output_size):
        super(RNN2, self).__init__()
        self.char_embeddings = torch.nn.Embedding(vocab_indexer_size, input_size)
        self.char_embeddings.weight.requires_grad = True

        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.v = torch.nn.Linear(hidden_size, int(hidden_size / 2))
        self.w = torch.nn.Linear(int(hidden_size / 2), output_size)

        self.loss = torch.nn.CrossEntropyLoss()

        torch.nn.init.xavier_uniform_(self.v.weight)
        torch.nn.init.xavier_uniform_(self.w.weight)

    def forward(self, x):
        x = self.char_embeddings(x)  # (batch_size, sequence length, embedding_size)
        output, h_n = self.gru(x)  # (num_layers  * directions (1), batch_size, hidden_size)
        h_n = torch.squeeze(h_n)  # (batch_size, hidden_size)
        return self.w(self.v(output)), self.w(self.v(h_n))
