import json
from typing import Type

from full_stack_transformer.core.data.text_lines_parsers import TextLinesParser
from full_stack_transformer.core.text_input import TextInput


class JsonLinesParser(TextLinesParser):
    def __init__(self, text_input_cls: Type[TextInput]):
        self._cls = text_input_cls

    def parse(self, text: str) -> TextInput:
        inp = self._cls(**json.loads(text))
        return inp
