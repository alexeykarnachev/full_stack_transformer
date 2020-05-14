from dataclasses import dataclass
from typing import Optional

import torch


class ModelInputError(Exception):
    pass


@dataclass
class ModelInput(dict):
    """Base data container which represents model input."""
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
            raise ModelInputError(
                '`input_ids` and `token_type_ids` shapes must be equal.'
            )
        elif self.lm_labels is not None and \
                self.lm_labels.size() != self.input_ids.size():
            raise ModelInputError(
                '`lm_labels` were passed, bu there shape does not match '
                '`input_ids` shape.'
            )
        elif self.past is not None and self.input_ids.size()[1] != 1:
            raise ModelInputError(
                'If `past` passed, `input_ids` must contain only one token, '
                'i.e sequence length must be equal to 1.'
            )

    def cuda(self, gpu_id):
        """Sends object field to cuda."""
        fields = self.__dict__

        for name, field in fields.items():
            if hasattr(field, 'cuda'):
                self.__dict__[name] = field.cuda(gpu_id)

        return self

    def to(self, device):
        fields = self.__dict__

        for name, field in fields.items():
            if hasattr(field, 'to'):
                self.__dict__[name] = field.to(device)

        return self
