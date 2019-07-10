import os
import pickle

import torch
from gensim.models import Word2Vec
from torch import nn

import settings
from pre_processing import _create_dialog_pairs, UNK, _question_answer_datasets


class WordEmbedding:

    def __init__(self, **kwargs):
        self.freeze = False if 'freeze' not in kwargs.keys() else bool(kwargs.get('freeze'))
        self._embedding, self.pairs = self.load(**kwargs)
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

        if 'source' not in kwargs.keys():
            raise KeyError('Embedding should have directory_from or pairs attributed defined.')

        if isinstance(kwargs.get('source'), list):
            x, y = _question_answer_datasets(kwargs.get('source'))

            word2vec = Word2Vec(min_count=1, size=300, alpha=0.001, workers=settings.cores)
            word2vec.build_vocab(x + y)

            return word2vec, kwargs.get('source')

        if os.path.isfile(kwargs.get('source')):
            pairs = pickle.load(open(kwargs.get('source') + '.pairs.pickle', 'rb'))
            embeddings = Word2Vec.load(kwargs.get('source'))
            return embeddings, pairs

        raise ValueError('source attribute should be a iterable or filename')

    def save(self, target_folder, filename):
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        self._embedding.save(os.path.join(target_folder, filename))
        pickle.dump(self.pairs, open(os.path.join(target_folder, filename + ".pairs.pickle"), 'wb'))
        return self

    def train(self, *args, **kwargs):
        x, y = _create_dialog_pairs(self.pairs)

        epochs = 1000 if 'epochs' not in kwargs.keys() else kwargs.get('epochs')

        self._embedding.build_vocab(x + y, update=True)

        self._embedding.train(x + y, total_examples=self._embedding.corpus_count, epochs=epochs)

        return self
