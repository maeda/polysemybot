import os
import unittest
import tempfile
import uuid

import settings
import app
from embeddings import WordEmbeddingBasic, WordEmbeddingPreTrained

os.environ['BASE_DIR'] = tempfile.gettempdir()

from model import Model, EncoderRNN, DecoderRNN
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
        cls.word_embedding = WordEmbeddingBasic(pairs=cls.dataset.pairs)

        encoder = EncoderRNN(cls.word_embedding, 300, 1).to(settings.device)
        decoder = DecoderRNN(300, cls.word_embedding, 0.0, 1).to(settings.device)
        cls.model = Model(encoder, decoder)
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
        word_embedding = WordEmbeddingBasic(freeze=False, pairs=dataset.pairs)
        word_embedding.train()
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
        cls.dataset = ds.process(PreProcessing(sentences))

    def test_train(self):
        word_embedding = WordEmbeddingBasic(pairs=self.__class__.dataset.pairs)
        self.assertEqual(word_embedding.n_words(), 25)

    def test_load_from_file(self):
        embeddings_path = os.path.join(settings.BASE_DIR, 'embeddings', uuid.uuid4().hex)
        filename = str(self.__class__.dataset.idx) + ".bin"

        word_embedding = WordEmbeddingBasic(pairs=self.__class__.dataset.pairs)
        word_embedding.train()
        word_embedding.save(embeddings_path, filename)

        model = WordEmbeddingPreTrained(directory_from=os.path.join(embeddings_path, filename))
        print(model._embedding.wv.similarity('batendo', 'porta'))


class MainTest(unittest.TestCase):
    def test_main(self):
        app.run(hidden=300,
                layer=1,
                dropout=0.0,
                learning_rate=0.01,
                iteration=5,
                save=10,
                train='./data/starwars.txt',
                test=False)
