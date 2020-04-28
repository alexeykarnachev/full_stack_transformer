import abc
from typing import Callable

import tokenizers


class Tokenizer(abc.ABC, tokenizers.SentencePieceBPETokenizer):
    @staticmethod
    @abc.abstractmethod
    def get_pad_val() -> int:
        pass

    @staticmethod
    @abc.abstractmethod
    def get_preprocessor() -> Callable[[str], str]:
        pass

    @staticmethod
    @abc.abstractmethod
    def get_postprocessor() -> Callable[[str], str]:
        pass
