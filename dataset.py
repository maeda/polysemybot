import json
import os
import random
from typing import Generator

import settings
import torch
from nltk import word_tokenize

from pre_processing import PreProcessing
from utils import TensorHelper

SOS_token = 0
EOS_token = 1

SOS = "SOS"
EOS = "EOS"
UNK = "UNK"


def control_words():
    return [SOS, EOS, UNK]


class Vocabulary:
    def __init__(self):
        self._word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0
        self.add_word(SOS)
        self.add_word(EOS)
        self.add_word(UNK)

    def add_sentence_tuple(self, sentence_tuple: tuple):
        for sentence in sentence_tuple:
            self.add_sentence(sentence)

    def add_sentence(self, sentence):
        for word in word_tokenize(sentence):
            self.add_word(word)

    def add_word(self, word):
        if word not in self._word2index:
            self._word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def word2index(self, word):
        return self._word2index[UNK] if word not in self._word2index else self._word2index[word]


class Dataset:
    def __init__(self,
                 vocabulary: Vocabulary,
                 pairs: list,
                 idx: str,
                 tensor_helper: TensorHelper = TensorHelper(settings.device, EOS_token)):
        self.vocabulary = vocabulary
        self.pairs = pairs
        self.idx = idx
        self.tensor_helper = tensor_helper

    def vocab_size(self):
        return self.vocabulary.n_words

    def training_pairs(self, sample_size):
        return [self.tensor_helper.tensors_from_pair(random.choice(self.pairs), self.vocabulary, self.vocabulary)
                for _ in range(sample_size)]

    def __str__(self):
        return json.dumps({
            'idx': self.idx,
            'vocab': self.vocabulary.n_words,
            'pairs': len(self.pairs)
        })


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

    return Dataset(vocabulary, pairs, reader.idx)


def save(dataset: Dataset):
    directory = os.path.join(settings.TRAINING_DATA_DIR, dataset.idx)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(dataset.vocabulary, os.path.join(directory, '{!s}.torch'.format('vocab')))
    torch.save(dataset.pairs, os.path.join(directory, '{!s}.torch'.format('pairs')))


def load(dataset_id: str):
    directory = os.path.join(settings.TRAINING_DATA_DIR, dataset_id)
    vocab = torch.load(os.path.join(directory, '{!s}.torch'.format('vocab')))
    pairs = torch.load(os.path.join(directory, '{!s}.torch'.format('pairs')))

    return Dataset(vocab, pairs, dataset_id)

