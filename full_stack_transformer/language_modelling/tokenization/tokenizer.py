import abc
import json
import re
from typing import List

import numpy as np
from tokenizers.implementations import BaseTokenizer
from transformers.tokenization_utils import PreTrainedTokenizerFast

from full_stack_transformer.language_modelling.data_structures import (
    Document,
    DocumentEncoding
)
from full_stack_transformer.utilities.factory import get_object

_END_OF_DOCUMENT = '[END_OF_DOCUMENT]'
_START_OF_DOCUMENT = '[START_OF_DOCUMENT]'
_END_OF_META = '[END_OF_META]'

_SPECIAL_TOKENS = [_END_OF_DOCUMENT, _START_OF_DOCUMENT, _END_OF_META]

LOSS_IGNORE = -100


class DocumentTokenizer(PreTrainedTokenizerFast):
    @property
    def eos_token_id(self):
        return self.convert_tokens_to_ids(_END_OF_DOCUMENT)

    @property
    def vocab_size(self):
        return max(self.all_special_ids) + 1

    def __init__(
            self,
            tokenizer: BaseTokenizer,
            max_meta_len: int,
            max_body_len: int,
            ignore_meta_prob: float = 0,
            **kwargs
    ):
        super().__init__(tokenizer, **kwargs)

        self._max_meta_len = max_meta_len
        self._max_body_len = max_body_len
        self._ignore_meta_prob = ignore_meta_prob
        self.add_special_tokens({'additional_special_tokens': _SPECIAL_TOKENS})

    def encode_document(
            self,
            document: Document,
            with_eos: bool = True
    ) -> DocumentEncoding:

        body = document.body
        if np.random.rand() > self._ignore_meta_prob:
            meta = document.meta
        else:
            meta = None

        body_ids, body_lm_labels = self._get_body_ids_and_labels(
            body=body,
            with_eos=with_eos
        )
        meta_ids, meta_lm_labels = self._get_meta_ids_and_labels(
            meta=meta
        )

        encoding = self._get_encoding_from_ids_and_labels(
            body_ids=body_ids,
            body_lm_labels=body_lm_labels,
            meta_ids=meta_ids,
            meta_lm_labels=meta_lm_labels
        )

        return encoding

    def decode_encoding(self, encoding: DocumentEncoding) -> str:
        toke_ids = encoding.token_ids
        text = self.decode(token_ids=toke_ids, skip_special_tokens=True)
        text = self.postrpocess_decoded(text)
        return text

    def _get_body_ids_and_labels(self, body: str, with_eos: bool):
        body = self.prepare_for_tokenization(text=body)
        body = f'{_START_OF_DOCUMENT}{body}'

        if with_eos:
            body += _END_OF_DOCUMENT

        body_ids = self.encode(text=body)
        body_lm_labels = list(body_ids)
        body_lm_labels[0] = LOSS_IGNORE

        return body_ids, body_lm_labels

    def _get_meta_ids_and_labels(self, meta: str):
        if meta is not None:
            meta = self.prepare_for_tokenization(text=meta)
            meta = f'{meta}{_END_OF_META}'
            meta_ids = self.encode(text=meta)
            meta_ids = meta_ids[-self._max_meta_len:]
            meta_lm_labels = [LOSS_IGNORE] * len(meta_ids)
        else:
            meta_ids = []
            meta_lm_labels = []

        return meta_ids, meta_lm_labels

    def _get_encoding_from_ids_and_labels(
            self,
            body_ids: List[int],
            body_lm_labels: List[int],
            meta_ids: List[int],
            meta_lm_labels: List[int]
    ) -> DocumentEncoding:
        token_ids = meta_ids + body_ids[:self._max_body_len]
        lm_labels = meta_lm_labels + body_lm_labels[:self._max_body_len]
        encoding = DocumentEncoding(
            token_ids=token_ids,
            lm_labels=lm_labels
        )

        return encoding

    def encode_line(self, line: str) -> DocumentEncoding:
        document = Document(**json.loads(line))
        encoding = self.encode_document(document=document)
        return encoding

    @abc.abstractmethod
    def prepare_for_tokenization(self, text: str) -> str:
        pass

    def postrpocess_decoded(self, text: str) -> str:
        for token in _SPECIAL_TOKENS:
            text = re.sub(re.escape(token), '', text)

        processed = self._postrpocess_decoded(text=text)

        if processed is not None:
            text = processed

        text = text.strip()
        return text

    @abc.abstractmethod
    def _postrpocess_decoded(self, text: str) -> str:
        pass


def get_tokenizer(
        name: str,
        max_meta_len: int,
        max_body_len: int,
        ignore_meta_prob: float
) -> DocumentTokenizer:
    path = f'full_stack_transformer.language_modelling.tokenization.{name}'
    tokenizer = get_object(
        class_path=path,
        max_meta_len=max_meta_len,
        max_body_len=max_body_len,
        ignore_meta_prob=ignore_meta_prob
    )

    if not isinstance(tokenizer, DocumentTokenizer):
        raise ValueError(f'{name} is not a `DocumentTokenizer` subclass.')

    return tokenizer
