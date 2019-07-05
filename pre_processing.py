import re
import uuid

from typing import Generator, Iterable

from nltk import tokenize

SOS = "SOS"
EOS = "EOS"
UNK = "UNK"


def control_words():
    return [SOS, EOS, UNK]


def _create_dialog_pairs(lines) -> Generator:
    iterator = iter(lines)

    current_item = next(iterator)

    for item in iterator:
        yield (current_item, item)
        current_item = item


def _question_answer_datasets(pairs):
    x = []
    y = []
    for pair in pairs:
        x.append([SOS] + tokenize.word_tokenize(str(pair[0]).lower(), language='portuguese') + [UNK] + [EOS])
        y.append([SOS] + tokenize.word_tokenize(str(pair[1]).lower(), language='portuguese') + [UNK] + [EOS])

    return x, y


class PreProcessing:

    def __init__(self, reader: Iterable, idx: str = uuid.uuid4().hex):
        self.reader = reader
        self.idx = idx

    def __iter__(self) -> Generator:
        for item in (item for item in self.reader if item):
            result = self._process(item)

            if not result:
                continue

            yield result

    def _process(self, text):
        text = self._normalize_string(text)
        # add here other wrangling steps...
        return text

    def _normalize_string(self, s):
        s = s.lower().strip()
        s = re.sub(r"([.!?])", r" \1", s)
        # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def process(self):
        return list(_create_dialog_pairs(self))
