from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor]


@dataclass
class LanguageModelOutput(ModelOutput):
    hidden: torch.Tensor
    past: torch.Tensor
    lm_loss: Optional[torch.Tensor] = None
    ul_loss: Optional[torch.Tensor] = None
