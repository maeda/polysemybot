from __future__ import unicode_literals, print_function, division

from typing import Generator

from pre_processing import PreProcessing

SOS_token = 0
EOS_token = 1


class Vocabulary:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence_tuple(self, sentence_tuple: tuple):
        for sentence in sentence_tuple:
            self.add_sentence(sentence)

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class Dataset:
    def __init__(self, vocabulary, pairs):
        self.vocabulary = vocabulary
        self.pairs = pairs

    def vocab_size(self):
        return self.vocabulary.n_words


def process(reader: PreProcessing):

    def _create_dialog_pairs(lines) -> Generator:
        iterator = iter(lines)

        current_item = next(iterator)

        for item in iterator:
            yield (current_item, item)
            current_item = item

    pairs = list(_create_dialog_pairs(reader))
    vocabulary = Vocabulary()

    print("Counting words...")
    for pair in pairs:
        vocabulary.add_sentence_tuple(pair)

    print("Counted words:")
    print(vocabulary.n_words)

    return Dataset(vocabulary, pairs)





