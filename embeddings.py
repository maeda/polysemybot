import errno
import os
from abc import ABC, abstractmethod

import torch
from gensim.models import Word2Vec
from torch import nn

import settings
from pre_processing import _create_dialog_pairs, UNK, _question_answer_datasets


class WordEmbedding(ABC):

    def __init__(self, **kwargs):
        self.freeze = False if 'freeze' not in kwargs.keys() else bool(kwargs.get('freeze'))
        self._embedding = self.load(**kwargs)
        self.embedding_layer = self._build_layer()

    def n_words(self):
        return len(self._embedding.wv.vocab)

    def word2index(self, word):
        return self._embedding.wv.vocab[UNK].index if word not in self._embedding.wv.vocab else self._embedding.wv.vocab[word].index

    def index2word(self, idx):
        return self._embedding.wv.index2word[idx]

    def _build_layer(self) -> nn.Embedding:
        weights = torch.FloatTensor(self._embedding.wv.vectors)

        return nn.Embedding.from_pretrained(weights, freeze=self.freeze)

    def load(self, **kwargs):
        directory_from = kwargs['directory_from']
        if not os.path.isfile(directory_from):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), directory_from)

        return Word2Vec.load(directory_from)

    def save(self, target_folder, filename):
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        self._embedding.save(os.path.join(target_folder, filename))
        return self

    @abstractmethod
    def train(self, *args, **kwargs):
        pass


class WordEmbeddingBasic(WordEmbedding):
    def __init__(self, **kwargs):
        if 'pairs' not in kwargs.keys():
            raise KeyError("pairs is required")

        self.pairs = kwargs['pairs']
        super().__init__(**kwargs)

    def load(self, **kwargs):
        print("Counting words...")
        x, y = _question_answer_datasets(self.pairs)

        word2vec = Word2Vec(min_count=1, size=300, alpha=0.001, workers=settings.cores)
        word2vec.build_vocab(x + y)

        return word2vec

    def train(self, epochs=1000):
        x, y = _create_dialog_pairs(self.pairs)

        self._embedding.build_vocab(x + y, update=True)

        self._embedding.train(x + y, total_examples=self._embedding.corpus_count, epochs=epochs)

        return self


class WordEmbeddingPreTrained(WordEmbedding):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self,
              pairs,
              min_count=1,
              size=300,
              alpha=0.001,
              workers=settings.cores,
              epochs=1000):

        x, y = _create_dialog_pairs(pairs)

        self._embedding.build_vocab(x + y, update=True)

        self._embedding.train(x + y, total_examples=self._embedding.corpus_count, epochs=epochs)
