from __future__ import unicode_literals, print_function, division
import re

SOS_token = 0
EOS_token = 1


class Speaker:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def _normalize_string(s, preprocess=lambda s: s):
    s = preprocess(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


class DatasetReader:

    def __init__(self, dataset):
        self.dataset = dataset

    @staticmethod
    def _create_dialog_pairs(lines, normalize):
        iterator = iter(lines)

        current_item = next(iterator)

        for item in iterator:
            yield (normalize(current_item), normalize(item))
            current_item = item

    def read_utterances(self, normalize=lambda s: s):
        print("Reading lines...")

        pairs = [item for item in self._create_dialog_pairs(self.dataset, normalize)]

        input_speaker = Speaker("input_speaker")
        output_speaker = Speaker("output_speaker")

        return input_speaker, output_speaker, pairs


class PreProcessing:

    def __init__(self, dataset_reader: DatasetReader):
        self._dataset_reader = dataset_reader

    def prepare_data(self):
        input_speaker, output_speaker, pairs = self._dataset_reader.read_utterances(_normalize_string)
        print("Counting words...")
        for pair in (item for item in pairs if len(item[0]) > 0 and len(item[1]) > 0):
            input_speaker.add_sentence(pair[0])
            output_speaker.add_sentence(pair[1])

        print("Counted words:")
        print(input_speaker.name, input_speaker.n_words)
        print(output_speaker.name, output_speaker.n_words)

        return input_speaker, output_speaker, pairs
