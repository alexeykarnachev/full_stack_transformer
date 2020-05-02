import abc
import re
from typing import Sequence

import tokenizers


class Tokenizer(abc.ABC, tokenizers.implementations.BaseTokenizer):
    @abc.abstractmethod
    def get_pad_token(self) -> str:
        pass

    @abc.abstractmethod
    def get_eos_token(self) -> str:
        pass

    @abc.abstractmethod
    def get_bos_token(self) -> str:
        pass

    def get_pad_token_id(self) -> int:
        return self.token_to_id(self.get_pad_token())

    def get_eos_token_id(self) -> int:
        return self.token_to_id(self.get_eos_token())

    def get_bos_token_id(self) -> int:
        return self.token_to_id(self.get_bos_token())

    def prepare_for_tokenization(
            self,
            string: str,
            add_bos: bool = False,
            add_eos: bool = False) -> str:
        string = string or ''
        string = self._preprocess(string)
        if add_bos:
            string = f'{self.get_bos_token()} {string}'

        if add_eos:
            string = f'{string} {self.get_eos_token()}'

        string = string.strip()

        return string

    def prepare_and_encode(
            self,
            string: str,
            add_bos: bool = False,
            add_eos: bool = False) -> tokenizers.Encoding:
        prepared_string = self.prepare_for_tokenization(
            string=string, add_bos=add_bos, add_eos=add_eos)

        return self.encode(prepared_string)

    def prepare_and_encode_batch(
            self,
            strings: Sequence[str],
            add_bos: bool = False,
            add_eos: bool = False) -> Sequence[tokenizers.Encoding]:
        prepared_strings = []
        for string in strings:
            prepared_string = self.prepare_for_tokenization(
                string=string, add_bos=add_bos, add_eos=add_eos)
            prepared_strings.append(prepared_string)

        return self.encode_batch(prepared_strings)

    @abc.abstractmethod
    def _preprocess(self, string: str) -> str:
        pass

    def clean_after_generation(self, string: str, remove_bos_eos: bool = False):
        if remove_bos_eos:
            bos_eos_pattern = f'{self.get_bos_token()}|{self.get_eos_token()}'
            string = re.sub(re.escape(bos_eos_pattern), '', string)

        string = self._postprocess(string)

        return string

    @abc.abstractmethod
    def _postprocess(self, sequence: str) -> str:
        pass
