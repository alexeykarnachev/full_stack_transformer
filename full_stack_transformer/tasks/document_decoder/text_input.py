from dataclasses import dataclass
from typing import Optional

from full_stack_transformer.core.text_input import TextInput


@dataclass
class DocumentInput(TextInput):
    body: str
    meta: Optional[str] = None
