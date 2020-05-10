from typing import Sequence

import torch

from full_stack_transformer.utilities.sequences import pad_sequences_from_right
from full_stack_transformer.language_modelling.data_structures import (
    DocumentEncoding,
    LanguageModelInput
)
from full_stack_transformer.language_modelling.tokenization.tokenizer import \
    LOSS_IGNORE


class DocumentEncodingsCollate:
    def __init__(self, pad_value: int):
        self._pad_value = pad_value

    def __call__(
            self,
            encodings: Sequence[DocumentEncoding]
    ) -> LanguageModelInput:
        model_input = _collate_encodings(
            encodings=encodings,
            pad_value=self._pad_value
        )

        return model_input


def _collate_encodings(
        encodings: Sequence[DocumentEncoding],
        pad_value: int
) -> LanguageModelInput:
    token_ids = pad_sequences_from_right(
        sequences=[e.token_ids for e in encodings],
        pad_value=pad_value,
        max_len=None,
    )
    lm_labels = pad_sequences_from_right(
        sequences=[e.lm_labels for e in encodings],
        pad_value=LOSS_IGNORE,
        max_len=None
    )

    input_ids = torch.tensor(token_ids, dtype=torch.long)
    lm_labels = torch.tensor(lm_labels, dtype=torch.long)

    model_input = LanguageModelInput(
        input_ids=input_ids,
        lm_labels=lm_labels
    )

    return model_input
