import abc
import re
from typing import List

from tokenizers.implementations import BaseTokenizer
from transformers.tokenization_utils import PreTrainedTokenizerFast

from full_stack_transformer.core.encoding import Encoding
from full_stack_transformer.core.text_input import TextInput

_END = '[END]'

_SPECIAL_TOKENS = [_END]


class Tokenizer(PreTrainedTokenizerFast):
    @property
    def eos_token_id(self):
        return self.convert_tokens_to_ids(self.eos_token)

    @property
    def eos_token(self):
        return _END

    @property
    def vocab_size(self):
        return max(self.all_special_ids) + 1

    def __init__(self, tokenizer: BaseTokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)

        self.add_special_tokens({'additional_special_tokens': _SPECIAL_TOKENS})

    def encode_for_train(self, text_input: TextInput) -> List[Encoding]:
        text_input = self._preprocess_input(text_input=text_input)
        encodings = self._encode_for_train(text_input=text_input)

        return encodings

    def encode_for_inference(self, text_input: TextInput) -> List[Encoding]:
        text_input = self._preprocess_input(text_input=text_input)
        encodings = self._encode_for_inference(text_input=text_input)

        return encodings

    @abc.abstractmethod
    def _encode_for_train(self, text_input: TextInput) -> List[Encoding]:
        pass

    @abc.abstractmethod
    def _encode_for_inference(self, text_input: TextInput) -> List[Encoding]:
        pass

    @abc.abstractmethod
    def _preprocess_input(self, text_input: TextInput) -> TextInput:
        pass

    @abc.abstractmethod
    def _postprocess_text(self, text: str) -> str:
        pass

    def decode_encoding(self, encoding: Encoding) -> str:
        token_ids = encoding.token_ids
        text = self.decode(token_ids=token_ids, skip_special_tokens=True)

        for token in _SPECIAL_TOKENS:
            text = re.sub(re.escape(token), '', text)

        text = self._postprocess_text(text)

        text = text.strip()

        return text
