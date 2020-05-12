import abc

import torch
import torch.nn as nn

from full_stack_transformer.core.model_input import ModelInput
from full_stack_transformer.core.model_output import ModelOutput


class Model(nn.Module):

    @property
    def device(self):
        return self.parameters().__next__().device

    def __init__(self):
        super().__init__()
        pass

    @abc.abstractmethod
    def forward(self, inp: ModelInput) -> ModelOutput:
        pass

    @torch.no_grad()
    @abc.abstractmethod
    def infer(self, inp: ModelInput) -> ModelOutput:
        pass
