import abc
import json
from typing import Sequence, List
import numpy as np
import more_itertools
from tokenizers.implementations import BaseTokenizer
from transformers.tokenization_utils import PreTrainedTokenizerFast

from full_stack_transformer.utilities.factory import get_object
from full_stack_transformer.language_modelling.data_structures import (
    Document,
    DocumentEncoding
)

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

    def encode_document(self, document: Document) -> Sequence[DocumentEncoding]:

        body = document.body
        if np.random.rand() > self._ignore_meta_prob:
            meta = document.meta
        else:
            meta = None

        body_ids, body_lm_labels = self._get_body_ids_and_labels(body=body)
        meta_ids, meta_lm_labels = self._get_meta_ids_and_labels(meta=meta)

        encodings = self._get_encodings_from_ids_and_labels(
            body_ids=body_ids,
            body_lm_labels=body_lm_labels,
            meta_ids=meta_ids,
            meta_lm_labels=meta_lm_labels
        )

        return encodings

    def _get_body_ids_and_labels(self, body: str):
        body = self.prepare_for_tokenization(text=body)
        body = f'{_START_OF_DOCUMENT}{body}{_END_OF_DOCUMENT}'
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

    def _get_encodings_from_ids_and_labels(
            self,
            body_ids: List[int],
            body_lm_labels: List[int],
            meta_ids: List[int],
            meta_lm_labels: List[int]
    ) -> Sequence[DocumentEncoding]:
        encodings = []
        chunks = more_itertools.chunked(
            iterable=zip(body_ids, body_lm_labels),
            n=self._max_body_len
        )

        for chunk in chunks:
            ids, labels = list(zip(*chunk))
            token_ids = meta_ids + list(ids)
            lm_labels = meta_lm_labels + list(labels)
            encoding = DocumentEncoding(
                token_ids=token_ids,
                lm_labels=lm_labels
            )
            encodings.append(encoding)

        return encodings

    def encode_line(self, line: str) -> Sequence[DocumentEncoding]:
        document = Document(**json.loads(line))
        encodings = self.encode_document(document=document)
        return encodings

    @abc.abstractmethod
    def prepare_for_tokenization(self, text) -> str:
        pass


def get_tokenizer(
        name: str,
        max_meta_len: int,
        max_body_len: int,
        ignore_meta_prob: float
):
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
