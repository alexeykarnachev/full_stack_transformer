from typing import List, Optional

from pydantic import BaseModel, Field


class GeneratedTexts(BaseModel):
    texts: List[str] = Field()


class LanguageGeneratorAppParams(BaseModel):
    body: str = Field(default='', max_length=1024)
    meta: Optional[str] = Field(default=None, max_length=1024)

    max_number_of_generated_tokens: int = Field(default=128, ge=1, le=512)
    temperature: float = Field(default=1.0, gt=0, le=100)
    top_k: int = Field(default=0, ge=0)
    top_p: float = Field(default=0.99, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=1.0, le=100)
    num_return_sequences: int = Field(default=1, ge=1, le=64)
