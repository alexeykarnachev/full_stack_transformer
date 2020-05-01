from fastapi import FastAPI

from lm_trainer.application.schemas import (
    GeneratedTexts,
    TextGeneratorAppParams)
from lm_trainer.text_generator.text_generator import (
    TextGenerator)


class ViewsRegister:
    def __init__(
            self,
            app: FastAPI,
            generator: TextGenerator):
        self._app = app
        self._generator = generator

    def register_generated_texts_view(self):
        @self._app.post("/generated_texts/", response_model=GeneratedTexts)
        def generated_texts(
                text_generator_params: TextGeneratorAppParams
        ) -> GeneratedTexts:
            texts = self._generator(text_generator_params)
            return GeneratedTexts(texts=texts)

        return generated_texts

    def register_all_views(self):
        self.register_generated_texts_view()
