import unittest

from model import Model
from dataset import process
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

        dataset = process(PreProcessing(sentences))

        self.assertEqual(dataset.vocab_size(), 25)


class ModelTest(unittest.TestCase):

    def test_train_model(self):

        dataset = process(PreProcessing(sentences))

        model = Model()
        model.train(dataset=dataset)
        model.evaluate_randomly(dataset=dataset)

        self.assertEqual(dataset.vocab_size(), 25)

    def test_predict(self):
        dataset = process(PreProcessing(sentences))

        model = Model()
        model.train(dataset=dataset)

        output_words = model.evaluate(dataset.vocabulary, "Oi!")

        print(' '.join(output_words))

