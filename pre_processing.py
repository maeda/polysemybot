import re
import uuid

from typing import Generator, Iterable


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