# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """

    def __init__(self, word_embeddings: WordEmbeddings, input_size: int, hidden_size: int, output_size: int):
        SentimentClassifier.__init__(self)
        self.word_embeddings = word_embeddings
        self.indexer = self.word_embeddings.word_indexer
        self.model = FFNN(word_embeddings, input_size, hidden_size, output_size)

    def predict(self, ex_words: List[str]):
        word_indices = []
        for word in ex_words:
            if self.indexer.index_of(word) > 1:
                word_indices.append(self.indexer.index_of(word))
            else:
                word_indices.append(1)
        words_indices_tensor = torch.tensor([word_indices])
        probs = self.model.forward(words_indices_tensor)
        return torch.argmax(probs)


# Support function: takes a batch containing a list of sentences and maps all words to indices through the Indexer

def mapping(batch_ex_words: List[List], model: NeuralSentimentClassifier):
    max_len = 0
    final_list = []
    for i in range(len(batch_ex_words)):
        tmp = []
        for word in batch_ex_words[i]:
            if model.indexer.index_of(word) > 1:
                tmp.append(model.indexer.index_of(word))
            else:
                tmp.append(1)
        if len(tmp) > max_len:
            max_len = len(tmp)
        final_list.append(tmp)
    return final_list, max_len


# Support function: takes a batch and converts all of its sentences to max_len through padding (zeroes at the end)

def padding(batch_ex_word_indexer_values: List[List], max_len: int):
    for i in range(len(batch_ex_word_indexer_values)):
        item = batch_ex_word_indexer_values[i]
        if len(item) < max_len:
            while len(item) < max_len:
                item.append(0)
        batch_ex_word_indexer_values[i] = item
    return batch_ex_word_indexer_values


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """

    # Declare parameters

    input_size = word_embeddings.get_embedding_length()
    hidden_size = 32
    output_size = 2

    ffnn = NeuralSentimentClassifier(word_embeddings, input_size, hidden_size, output_size)

    num_epochs = 20
    batch_size = 128

    lr = 0.005  # 0.001
    optimizer = optim.Adam(ffnn.model.parameters(), lr=lr)

    # Generate an index for each SentimentExample item of the train_exs list.
    dataset_ex_indices = [i for i in range(0, len(train_exs))]

    for epoch in range(0, num_epochs):
        print("Epoch: %d" % epoch)
        # New Epoch - Shuffle the data and adjust learning rate accordingly
        if epoch % 10 == 0:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / (epoch + 1)
        random.shuffle(dataset_ex_indices)
        total_loss = 0.0

        for i in range(0, len(dataset_ex_indices), batch_size):
            # Select a batch of indices from the dataset, of size equal to the batch_size parameter
            batch_ex_indices = dataset_ex_indices[i: i + batch_size]

            # For each SentimentExample item of that batch, get the words and the label that correspond to it
            # Thus, creating two lists for the batch: a list of sentences (list of lists of words) and a list of labels
            batch_ex_words = []
            batch_ex_labels = []
            for j in range(len(batch_ex_indices)):
                batch_ex_words.append(train_exs[batch_ex_indices[j]].words)
                batch_ex_labels.append(train_exs[batch_ex_indices[j]].label)

            # Map words to indices through the Indexer
            # Thus transforming the list of lists of words to a list of lists of mapped integer values
            # Also, calculate the maximum sentence (list of words) length of the particular batch
            batch_ex_word_indexer_values, max_len = mapping(batch_ex_words, ffnn)

            # Conduct padding so all elements of the list are equal to the maximum length
            batch_ex_word_indexer_values = padding(batch_ex_word_indexer_values, max_len)

            # Print an example and make sure all of the lists are of size equal to the batch_size parameter
            # print(batch_ex_indices[0], batch_ex_words[0], batch_ex_word_indexer_values[0], batch_ex_labels[0])
            # print(len(batch_ex_indices), len(batch_ex_words), len(batch_ex_word_indexer_values), len(batch_ex_labels))

            # Transform the lists to tensors
            tensor_x = torch.tensor(batch_ex_word_indexer_values)
            tensor_y = torch.tensor(batch_ex_labels)

            # Train the model as per the lectures and calculate the total loss for the corresponding epoch
            ffnn.model.train()
            ffnn.model.zero_grad()
            probs = ffnn.model.forward(tensor_x)
            loss = ffnn.model.loss(probs, tensor_y)
            total_loss += loss
            loss.backward()
            optimizer.step()
        print("Total loss on epoch %i: %f" % (epoch, total_loss))

    return ffnn


# The feedforward neural network architecture as per the lectures
# We get the pretrained embeddings but set word_embeddings.weight.requires_grad = True to further fit them to our data
# The selected loss function is CrossEntropyLoss
class FFNN(nn.Module):
    def __init__(self, word_embeddings, input_size, hidden_size, output_size):
        super(FFNN, self).__init__()

        self.word_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(word_embeddings.vectors))
        self.word_embeddings.weight.requires_grad = True

        self.V1 = nn.Linear(input_size, hidden_size)
        self.g1 = nn.ReLU()
        self.d = nn.Dropout(0.1)
        self.V2 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.g2 = nn.ReLU()
        self.W = nn.Linear(int(hidden_size / 2), output_size)

        self.loss = nn.CrossEntropyLoss()

        nn.init.xavier_uniform_(self.V1.weight)
        nn.init.xavier_uniform_(self.V2.weight)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, x):
        x = self.word_embeddings(x)
        x = torch.mean(x, dim=1).float()  # (batch_size, L, embedding_size) -> (batch_size, embedding_size)
        return self.W(self.g2(self.V2(self.d((self.g1(self.V1(x)))))))
