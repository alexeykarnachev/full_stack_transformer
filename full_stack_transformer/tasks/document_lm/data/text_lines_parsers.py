import json
from typing import Optional

from full_stack_transformer.core.data.text_lines_parsers import TextLinesParser
from full_stack_transformer.tasks.document_lm.text_input import DocumentInput


class DocumentLinesParser(TextLinesParser):
    def __init__(self):
        pass

    def parse(self, text: str) -> Optional[DocumentInput]:
        inp = DocumentInput(**json.loads(text))
        return inp
