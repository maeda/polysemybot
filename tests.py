import os
import unittest
import tempfile

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

        dataset = ds.process(PreProcessing(sentences))

        self.assertEqual(dataset.vocab_size(), 25)


class ModelTest(unittest.TestCase):

    def test_train_model(self):

        dataset = ds.process(PreProcessing(sentences))

        model = Model(dataset.vocab_size(), dataset.vocab_size())
        model.train(dataset=dataset)
        model.evaluate_randomly(dataset=dataset)

        self.assertEqual(dataset.vocab_size(), 25)

    def test_predict(self):
        dataset = ds.process(PreProcessing(sentences))

        model = Model(dataset.vocab_size(), dataset.vocab_size())
        model.train(dataset=dataset)

        output_words = model.evaluate(dataset.vocabulary, "Oi!")

        self.assertTrue(len(' '.join(output_words)) > 0)

    def test_save(self):
        dataset = ds.process(PreProcessing(sentences))

        model = Model(dataset.vocab_size(), dataset.vocab_size())
        model.train(dataset=dataset)


class DatasetTest(unittest.TestCase):

    def test_should_save_load_dataset(self):
        storage = ds.DatasetStorage()
        expected = storage.save(ds.Dataset.build(PreProcessing(sentences)))

        result = storage.load(expected.idx)

        self.assertEqual('{"idx": "'+expected.idx+'", "vocab": 25, "pairs": 3}', result.__str__())

    def test_should_generate_training_pairs(self):
        dataset = ds.process(PreProcessing(sentences))
        self.assertEqual(len(dataset.training_pairs(2)), 2)

    def test_should_create_dataset_dir(self):
        storage = ds.DatasetStorage()
        dataset = ds.process(PreProcessing(sentences))

        self.assertTrue(storage.exist(dataset.idx))
