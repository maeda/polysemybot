import unittest

from model import Model
from preprocessing import DatasetReader, PreProcessing

sentences = [
    "Ontem à noite e anteontem à noite...",
    "Tommyknockers, Tommyknockers batendo na porta.",
    "Eu quero sair, não sei se posso... tenho medo do Tommyknockers",
    "Bobbi."
]


class PreProcessingTest(unittest.TestCase):

    def test_read_utterances(self):
        reader = DatasetReader(sentences)
        expected = [
            ("Ontem à noite e anteontem à noite...", "Tommyknockers, Tommyknockers batendo na porta."),
            ("Tommyknockers, Tommyknockers batendo na porta.", "Eu quero sair, não sei se posso... tenho medo do Tommyknockers"),
            ("Eu quero sair, não sei se posso... tenho medo do Tommyknockers", "Bobbi.")]
        input_speaker, output_speaker, lines = reader.read_utterances()

        self.assertEqual(expected, list(lines))

    def test_pre_processing(self):
        pre_processing = PreProcessing(DatasetReader(sentences))
        input_speaker, output_speaker, pairs = pre_processing.prepare_data()

        self.assertEqual(input_speaker.n_words, 23)
        self.assertEqual(output_speaker.n_words, 19)


class ModelTest(unittest.TestCase):

    def test_train_model(self):
        model = Model()
        model.train(PreProcessing(DatasetReader(sentences)))
        model.evaluate_randomly()
