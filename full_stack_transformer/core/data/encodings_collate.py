from typing import Sequence, Union, Optional

import torch

from full_stack_transformer.core.constants import LOSS_IGNORE
from full_stack_transformer.core.encoding import Encoding
from full_stack_transformer.core.model_input import ModelInput
from full_stack_transformer.utilities.sequences import pad_sequences_from_right


class EncodingsCollate:
    def __init__(self, pad_value: int):
        self._pad_value = pad_value

    def __call__(
            self,
            encodings: Sequence[Encoding],
            device: Optional[Union[torch.device, str]] = None
    ) -> ModelInput:
        model_input = _collate_encodings(
            encodings=encodings,
            pad_value=self._pad_value,
            device=device
        )

        return model_input


def _collate_encodings(
        encodings: Sequence[Encoding],
        pad_value: int,
        device: Optional[Union[torch.device, str]] = None
) -> ModelInput:
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

    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
    lm_labels = torch.tensor(lm_labels, dtype=torch.long, device=device)
    model_input = ModelInput(input_ids=input_ids, lm_labels=lm_labels)

    return model_input
