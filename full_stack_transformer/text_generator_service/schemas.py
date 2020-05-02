from typing import List, Sequence, Optional

from pydantic import BaseModel, Field

from full_stack_transformer.text_generator.text_generator import TextGeneratorParams


class GeneratedTexts(BaseModel):
    texts: List[str] = Field()


class TextGeneratorAppParams(TextGeneratorParams):
    generation_max_len: int = Field(default=64, ge=1, le=512)
    temperature: float = Field(default=0.7, gt=0, le=20)
    top_k: int = Field(default=50, ge=0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=5.0, ge=1.0, le=20)
    num_return_sequences: int = Field(default=4, ge=1, le=64)
    seed_text: str = Field(default='', max_length=1024)
    ignored_words: Optional[Sequence[str]] = Field(default=[], max_length=1024)
