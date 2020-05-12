import abc
from typing import Optional

from full_stack_transformer.core.text_input import TextInput


class TextLinesParser:
    @abc.abstractmethod
    def parse(self, text: str) -> Optional[TextInput]:
        pass
