from typing import Optional, Sequence

from fastapi import FastAPI

from lm_trainer.application.schemas import (
    GeneratedTexts,
    TextGeneratorAppParams)
from lm_trainer.text_generator.text_generator import (
    TextGenerator)
from lm_trainer.tokenization import Tokenizer


class ViewsRegister:
    def __init__(
            self,
            app: FastAPI,
            generator: TextGenerator,
            tokenizer: Tokenizer,
            ignored_token_ids: Optional[Sequence[int]]):
        self._app = app
        self._generator = generator
        self._tokenizer = tokenizer
        self._ignored_token_ids = ignored_token_ids

    def register_generated_texts_view(self):
        @self._app.post("/generated_texts/", response_model=GeneratedTexts)
        def generated_texts(
                text_generator_params: TextGeneratorAppParams
        ) -> GeneratedTexts:
            texts = self._generator(text_generator_params)
            return GeneratedTexts(texts=texts)

        return generated_texts
