from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ModelOutput:
    """Base data container for model output tensors."""
    logits: torch.Tensor
    loss: Optional[torch.Tensor]


@dataclass
class LanguageModelOutput(ModelOutput):
    """Data container for language model output tensors."""
    hidden: torch.Tensor
    past: torch.Tensor
    lm_loss: Optional[torch.Tensor] = None
    ul_loss: Optional[torch.Tensor] = None
