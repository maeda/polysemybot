import os
import unittest
import tempfile
import uuid

import settings

os.environ['BASE_DIR'] = tempfile.gettempdir()

from model import Model
import dataset as ds
from pre_processing import PreProcessing


sentences = [
    "Ontem à noite e anteontem à noite...",
    "",
    "Tommyknockers, Tommyknockers batendo na porta.",
    "Eu quero sair, não sei se posso... tenho medo do Tommyknockers",
    "Bobbi."
]


class PreProcessingTest(unittest.TestCase):

    def test_pre_processing(self):
        pre_processing = PreProcessing(sentences)
        dataset = ds.process(pre_processing)

        expected = [
            ('ontem à noite e anteontem à noite . . .', 'tommyknockers, tommyknockers batendo na porta .'),
            ('tommyknockers, tommyknockers batendo na porta .', 'eu quero sair, não sei se posso . . . tenho medo do tommyknockers'),
            ('eu quero sair, não sei se posso . . . tenho medo do tommyknockers', 'bobbi .')
        ]

        self.assertEqual(dataset.pairs, expected)


class ModelTest(unittest.TestCase):

    model = None
    dataset = None
    word_embedding = None

    @classmethod
    def setUpClass(cls):
        cls.pre_processing = PreProcessing(sentences)
        cls.dataset = ds.process(cls.pre_processing)
        cls.word_embedding = ds.WordEmbedding.train(cls.dataset.pairs)
        cls.model = Model(cls.word_embedding)
        cls.model.train(cls.dataset)

    def test_train_model(self):
        self.__class__.model.evaluate_randomly(dataset=self.__class__.dataset)

        self.assertEqual(self.__class__.word_embedding.n_words(), 25)

    def test_predict(self):
        output_words = self.__class__.model.evaluate("Oi!", self.__class__.dataset)

        self.assertTrue(len(' '.join(output_words)) > 0)

    def test_load_saved_model(self):
        model = Model.load(self.__class__.dataset.idx)
        model.summary()


class DatasetTest(unittest.TestCase):

    def test_should_save_load_dataset(self):
        storage = ds.DatasetStorage()
        pre_processing = PreProcessing(sentences)
        dataset = ds.process(pre_processing)
        expected = storage.save(dataset)

        result = storage.load(expected.idx)

        self.assertEqual('{"idx": "'+expected.idx+'", "pairs": 3}', result.__str__())

    def test_should_generate_training_pairs(self):
        pre_processing = PreProcessing(sentences)
        dataset = ds.process(pre_processing)
        word_embedding = ds.WordEmbedding.train(dataset.pairs)

        self.assertEqual(len(dataset.training_pairs(2, word_embedding)), 2)

    def test_should_create_dataset_dir(self):
        storage = ds.DatasetStorage()
        pre_processing = PreProcessing(sentences)
        dataset = ds.process(pre_processing)

        self.assertTrue(storage.exist(dataset.idx))


class WordEmbeddingTest(unittest.TestCase):

    dataset = None

    @classmethod
    def setUpClass(cls):
        cls.pre_processing = PreProcessing(sentences)
        cls. dataset = ds.process(cls.pre_processing)

    def test_train(self):
        word_embedding = ds.WordEmbedding.train(self.__class__.dataset.pairs)
        self.assertEqual(word_embedding.n_words(), 25)

    def test_load_from_file(self):
        embeddings_path = os.path.join(settings.BASE_DIR, 'embeddings', uuid.uuid4().hex)
        ds.WordEmbedding.train(self.__class__.dataset.pairs).save(embeddings_path, str(self.__class__.dataset.idx) + ".bin")
        model = ds.WordEmbedding.load_from_file(os.path.join(embeddings_path, str(self.__class__.dataset.idx) + ".bin"))
        print(model._embedding.wv.similarity('batendo', 'porta'))
