import abc
import re

import tokenizers


class Tokenizer(abc.ABC, tokenizers.SentencePieceBPETokenizer):
    """`Link text <http://target>`_"""

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
            add_eos: bool = False
    ) -> str:
        string = self._preprocess(string)
        if add_bos:
            string = f'{self.get_bos_token()} {string}'

        if add_eos:
            string = f'{string} {self.get_eos_token()}'

        return string

    @abc.abstractmethod
    def _preprocess(self, string: str) -> str:
        pass

    def clean_after_generation(self, string: str, remove_bos_eos: bool = False):
        if remove_bos_eos:
            bos_eos_pattern = f'{self.get_bos_token()}|{self.get_eos_token()}'
            string = re.sub(bos_eos_pattern, '', string)

        string = self._postprocess(string)

        return string

    @abc.abstractmethod
    def _postprocess(self, sequence: str) -> str:
        pass
