import abc
from typing import Sequence

from full_stack_transformer.core.encoding import Encoding
from full_stack_transformer.core.model_input import ModelInput


class EncodingsCollate:
    @abc.abstractmethod
    def __call__(self, encodings: Sequence[Encoding]) -> ModelInput:
        pass
