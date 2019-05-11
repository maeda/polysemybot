import math
import time

import torch


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


class TensorHelper:

    def __init__(self, device, eos_token):
        self._device = device
        self._eos_token = eos_token

    def tensors_from_pair(self, pair, input_lang, output_lang):
        input_tensor = self.tensor_from_sentence(input_lang, pair[0])
        target_tensor = self.tensor_from_sentence(output_lang, pair[1])
        return input_tensor, target_tensor

    def tensor_from_sentence(self, lang, sentence):
        indexes = self._indexes_from_sentence(lang, sentence)
        indexes.append(self._eos_token)
        return torch.tensor(indexes, dtype=torch.long, device=self._device).view(-1, 1)

    def _indexes_from_sentence(self, lang, sentence):
        return [lang.word2index(word) for word in sentence.split(' ')]