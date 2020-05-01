from typing import List

from pydantic import BaseModel, Field


class SeedText(BaseModel):
    text: str = Field(default='', max_length=1024)


class GeneratedTexts(BaseModel):
    texts: List[str] = Field()


class TextGeneratorAppParams(BaseModel):
    generation_max_len: int = Field(default=64, ge=1, le=512)
    temperature: float = Field(default=0.7, gt=0, le=20)
    top_k: int = Field(default=50, ge=0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=5.0, ge=1.0, le=20)
    num_return_sequences: int = Field(default=4, ge=1, le=64)
