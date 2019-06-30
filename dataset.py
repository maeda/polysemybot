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

    @staticmethod
    def build(reader: PreProcessing):
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

    def __str__(self):
        return json.dumps({
            'idx': self.idx,
            'vocab': self.vocabulary.n_words,
            'pairs': len(self.pairs)
        })


class DatasetStorage:

    def __init__(self, dataset_dir=settings.TRAINING_DATA_DIR):
        self.dataset_dir = dataset_dir

    def _get_dataset_dir(self, dataset_id: str):
        return os.path.join(settings.BASE_DIR, dataset_id, self.dataset_dir)

    def exist(self, dataset_id: str):
        return os.path.exists(self._get_dataset_dir(dataset_id))

    def create_dataset_dir(self, dataset_id: str):
        dataset_dir = self._get_dataset_dir(dataset_id)

        if not self.exist(dataset_id):
            os.makedirs(dataset_dir)

        return dataset_dir

    def save(self, dataset: Dataset) -> Dataset:
        dataset_dir = self.create_dataset_dir(dataset.idx)
        torch.save(dataset.vocabulary, os.path.join(dataset_dir, '{!s}.torch'.format('vocab')))
        torch.save(dataset.pairs, os.path.join(dataset_dir, '{!s}.torch'.format('pairs')))

        return dataset

    def load(self, dataset_id: str) -> Dataset:
        directory = self._get_dataset_dir(dataset_id)
        vocab = torch.load(os.path.join(directory, '{!s}.torch'.format('vocab')))
        pairs = torch.load(os.path.join(directory, '{!s}.torch'.format('pairs')))

        return Dataset(vocab, pairs, dataset_id)


def process(reader: PreProcessing, storage: DatasetStorage = DatasetStorage()):

    if not storage.exist(reader.idx):
        dataset = Dataset.build(reader)
        storage.save(dataset)

    return storage.load(reader.idx)


def load(dataset_id: str, storage: DatasetStorage = DatasetStorage()) -> Dataset:
    return storage.load(dataset_id)
