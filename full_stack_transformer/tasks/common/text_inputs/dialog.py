from dataclasses import dataclass
from typing import Optional, Sequence

from full_stack_transformer.core.text_input import TextInput


@dataclass
class DialogInput(TextInput):
    utterances: Sequence[str]
    persona_0: Optional[str] = None
    persona_1: Optional[str] = None
    tags: Optional[str] = None
