import errno
import json
import os
import random

from gensim.models import Word2Vec
from nltk import tokenize
from torch import nn

import settings
import torch

from pre_processing import PreProcessing

SOS = "SOS"
EOS = "EOS"
UNK = "UNK"


def control_words():
    return [SOS, EOS, UNK]


class WordEmbedding:

    def __init__(self, embedding: Word2Vec):
        self._embedding = embedding

    def n_words(self):
        return len(self._embedding.wv.vocab)

    def word2index(self, word):
        return self._embedding.wv.vocab[UNK].index if word not in self._embedding.wv.vocab else self._embedding.wv.vocab[word].index

    def index2word(self, idx):
        return self._embedding.wv.index2word[idx]

    def embedding(self):
        weights = torch.FloatTensor(self._embedding.wv.vectors)

        return nn.Embedding.from_pretrained(weights, freeze=True)

    @staticmethod
    def load_from_file(embedding_file):
        if not os.path.isfile(embedding_file):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), embedding_file)

        model = Word2Vec.load(embedding_file)

        return WordEmbedding(model)

    @staticmethod
    def train(pairs, embedding_id, target_folder=None, embedding=None):
        print("Counting words...")
        x = []
        y = []

        for pair in pairs:

            x.append([SOS] + tokenize.word_tokenize(str(pair[0]).lower(), language='portuguese') + [UNK] + [EOS])
            y.append([SOS] + tokenize.word_tokenize(str(pair[1]).lower(), language='portuguese') + [UNK] + [EOS])

        word2vec = embedding if embedding else Word2Vec(min_count=1, size=300, alpha=0.001, workers=4)

        word2vec.build_vocab(x + y, progress_per=10000, update=True)

        word2vec.train(x + y, total_examples=word2vec.corpus_count, epochs=1, report_delay=1)

        word_embedding = WordEmbedding(word2vec)

        return word_embedding.save(target_folder, embedding_id) if target_folder else word_embedding

    def save(self, target_folder, filename):
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        self._embedding.save(os.path.join(target_folder, filename))
        return self


class Dataset:
    def __init__(self,
                 pairs: list,
                 idx: str):
        self.pairs = pairs
        self.idx = idx

    def training_pairs(self, sample_size, word_embedding: WordEmbedding):
        return [self.tensors_from_pair(random.choice(self.pairs), word_embedding)
                for _ in range(sample_size)]

    def tensors_from_pair(self, pair, word_embedding: WordEmbedding):
        input_tensor = self.tensor_from_sentence(word_embedding, pair[0])
        target_tensor = self.tensor_from_sentence(word_embedding, pair[1])
        return input_tensor, target_tensor

    def tensor_from_sentence(self, word_embedding: WordEmbedding, sentence):
        indexes = self._indexes_from_sentence(word_embedding, sentence)
        indexes.append(word_embedding.word2index(EOS))
        return torch.tensor(indexes, dtype=torch.long, device=settings.device).view(-1, 1)

    def _indexes_from_sentence(self, word_embedding: WordEmbedding, sentence):
        return [word_embedding.word2index(word) for word in tokenize.word_tokenize(sentence.lower(), language='portuguese')]

    def __str__(self):
        return json.dumps({
            'idx': self.idx,
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
        torch.save(dataset.pairs, os.path.join(dataset_dir, '{!s}.torch'.format('pairs')))

        return dataset

    def load(self, dataset_id: str) -> Dataset:
        directory = self._get_dataset_dir(dataset_id)
        pairs = torch.load(os.path.join(directory, '{!s}.torch'.format('pairs')))

        return Dataset(pairs, dataset_id)


def process(reader: PreProcessing,
            storage: DatasetStorage = DatasetStorage()):

    if not storage.exist(reader.idx):
        pairs = reader.process()
        dataset = Dataset(pairs, reader.idx)
        storage.save(dataset)

    return storage.load(reader.idx)


def load(dataset_id: str, storage: DatasetStorage = DatasetStorage()) -> Dataset:
    return storage.load(dataset_id)
