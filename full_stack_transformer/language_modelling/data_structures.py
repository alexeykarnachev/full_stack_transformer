from dataclasses import dataclass
from typing import Optional, Sequence

import torch


class LanguageModelInputError(Exception):
    pass


@dataclass
class Document:
    body: str
    meta: Optional[str] = None


@dataclass
class DocumentEncoding:
    token_ids: Sequence[int]
    lm_labels: Sequence[int]


@dataclass
class LanguageModelInput(dict):
    input_ids: torch.Tensor
    token_type_ids: Optional[torch.Tensor] = None
    lm_labels: Optional[torch.Tensor] = None
    past: Optional[torch.Tensor] = None

    def __post_init__(self):
        super().__init__(**self.__dict__)

        self._check_inputs_validity()

    def _check_inputs_validity(self):
        if self.token_type_ids is not None and \
                self.input_ids.size() != self.token_type_ids.size():
            raise LanguageModelInputError(
                '`input_ids` and `token_type_ids` shapes must be equal.'
            )
        elif self.lm_labels is not None and \
                self.lm_labels.size() != self.input_ids.size():
            raise LanguageModelInputError(
                '`lm_labels` were passed, bu there shape does not match '
                '`input_ids` shape.'
            )
        elif self.past is not None and self.input_ids.size()[1] != 1:
            raise LanguageModelInputError(
                'If `past` passed, `input_ids` must contain only one token, '
                'i.e sequence length must be equal to 1.'
            )

    def cuda(self, gpu_id):
        fields = self.__dict__

        for name, field in fields.items():
            if isinstance(field, torch.Tensor):
                self.__dict__[name] = field.cuda(gpu_id)

        return self


@dataclass
class LanguageModelOutput:
    logits: torch.Tensor
    hidden: torch.Tensor
    past: torch.Tensor
    lm_loss: Optional[torch.Tensor] = None
    ul_loss: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
