from dataclasses import dataclass
from typing import Optional, Sequence

from full_stack_transformer.core.text_input import TextInput


@dataclass
class DialogInput(TextInput):
    utterances: Sequence[str]
    persona_idx: int
    persona: str
    tags: str
