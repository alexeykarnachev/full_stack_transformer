from typing import List

import numpy as np
from tokenizers.implementations import BaseTokenizer

from full_stack_transformer.core.constants import LOSS_IGNORE
from full_stack_transformer.core.encoding import Encoding
from full_stack_transformer.core.tokenizer import Tokenizer
from full_stack_transformer.tasks.common.text_inputs.document import DocumentInput
from full_stack_transformer.utilities.factory import get_object

_START_OF_DOCUMENT = '[START_OF_DOCUMENT]'
_END_OF_META = '[END_OF_META]'

_SPECIAL_TOKENS = [_START_OF_DOCUMENT, _END_OF_META]


class DocumentTokenizer(Tokenizer):
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

    def _encode_for_train(
            self,
            text_input: DocumentInput
    ) -> List[Encoding]:
        return self._encode(document=text_input, with_eos=True)

    def _encode_for_inference(
            self,
            text_input: DocumentInput
    ) -> List[Encoding]:
        return self._encode(document=text_input, with_eos=False)

    def _encode(
            self,
            document: DocumentInput,
            with_eos: bool = True
    ) -> List[Encoding]:

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

        return [encoding]

    def _get_body_ids_and_labels(self, body: str, with_eos: bool):
        body = self.prepare_for_tokenization(text=body)
        body = f'{_START_OF_DOCUMENT}{body}'

        if with_eos:
            body += self.eos_token

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
    ) -> Encoding:
        token_ids = meta_ids + body_ids[:self._max_body_len]
        lm_labels = meta_lm_labels + body_lm_labels[:self._max_body_len]
        encoding = Encoding(
            token_ids=token_ids,
            lm_labels=lm_labels
        )

        return encoding
