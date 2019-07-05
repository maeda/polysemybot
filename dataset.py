import json
import os
import random

from nltk import tokenize

import settings
import torch

from pre_processing import PreProcessing, EOS


class Dataset:
    def __init__(self,
                 pairs: list,
                 idx: str):
        self.pairs = pairs
        self.idx = idx

    def training_pairs(self, sample_size, word_embedding):
        return [self.tensors_from_pair(random.choice(self.pairs), word_embedding)
                for _ in range(sample_size)]

    def tensors_from_pair(self, pair, word_embedding):
        input_tensor = self.tensor_from_sentence(word_embedding, pair[0])
        target_tensor = self.tensor_from_sentence(word_embedding, pair[1])
        return input_tensor, target_tensor

    def tensor_from_sentence(self, word_embedding, sentence):
        indexes = self._indexes_from_sentence(word_embedding, sentence)
        indexes.append(word_embedding.word2index(EOS))
        return torch.tensor(indexes, dtype=torch.long, device=settings.device).view(-1, 1)

    def _indexes_from_sentence(self, word_embedding, sentence):
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
