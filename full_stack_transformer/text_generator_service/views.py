import re
from typing import Mapping

from fastapi import FastAPI

from full_stack_transformer.text_generator.text_generator import TextGenerator
from full_stack_transformer.text_generator_service.schemas import \
    GeneratedTexts, TextGeneratorAppParams


class ViewsRegister:
    def __init__(
            self,
            app: FastAPI,
            generator: TextGenerator,
            version: Mapping):
        self._app = app
        self._generator = generator
        self._version = version

    def register_generated_texts_view(self):
        @self._app.post("/generated_texts/", response_model=GeneratedTexts)
        def generated_texts(
                text_generator_params: TextGeneratorAppParams
        ) -> GeneratedTexts:
            texts = self._generator(text_generator_params)
            return GeneratedTexts(texts=texts)

        return generated_texts

    def register_health_check_view(self):
        @self._app.get("/health_check/")
        def health_check():
            return "ok"

        return health_check

    def register_version_view(self):
        @self._app.get("/version/")
        def version():
            return self._version

        return version

    def register_all_views(self):
        for field in dir(self):
            if re.match('^register.+view$', field):
                getattr(self, field)()
