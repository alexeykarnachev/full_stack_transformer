import abc
from typing import Callable

import tokenizers


class Tokenizer(abc.ABC, tokenizers.SentencePieceBPETokenizer):
    @abc.abstractmethod
    def get_pad_token_id(self) -> int:
        pass

    @abc.abstractmethod
    def get_eos_token_id(self) -> int:
        pass

    @abc.abstractmethod
    def get_bos_token_id(self) -> int:
        pass

    @staticmethod
    @abc.abstractmethod
    def get_preprocessor() -> Callable[[str], str]:
        pass

    @staticmethod
    @abc.abstractmethod
    def get_postprocessor() -> Callable[[str], str]:
        pass
