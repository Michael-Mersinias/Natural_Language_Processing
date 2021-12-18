import torch

from . import utils


class LanguageModel(object):
    def predict_all(self, some_text):
        """
        Given some_text, predict the likelihoods of the next character for each substring from 0..i
        The resulting tensor is one element longer than the input, as it contains probabilities for all sub-strings
        including the first empty string (probability of the first character)

        :param some_text: A string containing characters in utils.vocab, may be an empty string!
        :return: torch.Tensor((len(utils.vocab), len(some_text)+1)) of log-probabilities
        """
        raise NotImplementedError('Abstract function LanguageModel.predict_all')

    def predict_next(self, some_text):
        """
        Given some_text, predict the likelihood of the next character

        :param some_text: A string containing characters in utils.vocab, may be an empty string!
        :return: a Tensor (len(utils.vocab)) of log-probabilities
        """
        return self.predict_all(some_text)[:, -1]


class Bigram(LanguageModel):
    """
    Implements a simple Bigram model. You can use this to compare your TCN to.
    The bigram, simply counts the occurrence of consecutive characters in transition, and chooses more frequent
    transitions more often. See https://en.wikipedia.org/wiki/Bigram .
    Use this to debug your `language.py` functions.
    """

    def __init__(self):
        from os import path
        self.first, self.transition = torch.load(path.join(path.dirname(path.abspath(__file__)), 'bigram.th'))

    def predict_all(self, some_text):
        return torch.cat((self.first[:, None], self.transition.t().matmul(utils.one_hot(some_text))), dim=1)


class AdjacentLanguageModel(LanguageModel):
    """
    A simple language model that favours adjacent characters.
    The first character is chosen uniformly at random.
    Use this to debug your `language.py` functions.
    """

    def predict_all(self, some_text):
        prob = 1e-3*torch.ones(len(utils.vocab), len(some_text)+1)
        if len(some_text):
            one_hot = utils.one_hot(some_text)
            prob[-1, 1:] += 0.5*one_hot[0]
            prob[:-1, 1:] += 0.5*one_hot[1:]
            prob[0, 1:] += 0.5*one_hot[-1]
            prob[1:, 1:] += 0.5*one_hot[:-1]
        return (prob/prob.sum(dim=0, keepdim=True)).log()


class TCN(torch.nn.Module, LanguageModel):
    class CausalConv1dBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, dilation):
            """
            Your code here.
            Implement a Causal convolution followed by a non-linearity (e.g. ReLU).
            Optionally, repeat this pattern a few times and add in a residual block
            :param in_channels: Conv1d parameter
            :param out_channels: Conv1d parameter
            :param kernel_size: Conv1d parameter
            :param dilation: Conv1d parameter
            """
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.ConstantPad1d(((kernel_size-1)*dilation, 0), 0),
                torch.nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation),
                torch.nn.ReLU(),
                torch.nn.ConstantPad1d(((kernel_size-1)*dilation, 0), 0),
                torch.nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation),
                torch.nn.ReLU())
            self.skip = lambda x: x
            if in_channels != out_channels:
                self.skip = torch.nn.Conv1d(in_channels, out_channels,1)

        def forward(self, x):
            return self.network(x) + self.skip(x)

    def __init__(self, layers=[50, 50, 50, 50], kernel_size=3):
        """
        Your code here

        Hint: The probability of the first character should be a parameter
        use torch.nn.Parameter to explicitly create it.
        """
        super().__init__()

        c = len(utils.vocab)
        blocks = []
        for i, l in enumerate(layers):
            blocks.append(self.CausalConv1dBlock(c, l, kernel_size, 2**i))
            c = l

        self.first = torch.nn.Parameter(torch.zeros(len(utils.vocab)))
        self.network = torch.nn.Sequential(*blocks)
        self.classifier = torch.nn.Conv1d(c, len(utils.vocab), 1)

    def forward(self, x):
        """
        Your code here
        Return the logit for the next character for prediction for any substring of x

        @input: torch.Tensor((B, vocab_size, L)) a batch of one-hot encodings
        @return torch.Tensor((B, vocab_size, L+1)) a batch of log-likelihoods or logits
        """
        if x.numel() == 0:
            return self.first[None, :, None].expand(x.size(0), -1, -1)
        logit = self.classifier(self.network(x))
        return torch.cat((self.first[None, :, None].expand(x.size(0), -1, -1), logit), dim=2)

    def predict_all(self, some_text):
        return torch.nn.functional.log_softmax(self.forward(utils.one_hot(some_text)[None])[0], dim=0)


def save_model(model):
    from os import path
    return torch.save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'tcn.th'))


def load_model():
    from os import path
    r = TCN()
    r.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'tcn.th'), map_location='cpu'))
    return r


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser("Compute the log-likelihood of some models")
    parser.add_argument('-m', '--model', choices=['Adjacent', 'Bigram', 'TCN'], default='Adjacent')
    args = parser.parse_args()

    lm = AdjacentLanguageModel() if args.model == 'Adjacent' else (load_model() if args.model == 'TCN' else Bigram())

    from .utils import SpeechDataset

    data = SpeechDataset('data/valid.txt', max_len=None)
    lls, norm_lls = [], []
    for d in data:
        p = lm.predict_all(d)
        ll = float((p[:, :-1]*utils.one_hot(d)).sum())
        lls.append(ll)
        norm_lls.append(ll/len(d))
    import numpy as np
    print('%s model' % args.model)
    print('  Average log-likelihood model              = ', np.mean(lls))
    print('  Average log-likelihood model (normalized) = ', np.mean(norm_lls))
