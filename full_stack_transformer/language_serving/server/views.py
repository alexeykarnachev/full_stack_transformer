import re
from typing import Mapping

from fastapi import FastAPI

from full_stack_transformer import LanguageGenerator, Document
from full_stack_transformer.language_modelling.tokenization.tokenizer import \
    DocumentTokenizer
from full_stack_transformer.language_serving.server.schemas import (
    GeneratedTexts,
    LanguageGeneratorAppParams
)


class ViewsRegister:
    def __init__(
            self,
            app: FastAPI,
            generator: LanguageGenerator,
            tokenizer: DocumentTokenizer,
            version: Mapping
    ):
        self._app = app
        self._generator = generator
        self._version = version
        self._tokenizer = tokenizer

    def register_generated_texts_view(self):
        @self._app.post("/generated_texts/", response_model=GeneratedTexts)
        def generated_texts(
                text_generator_params: LanguageGeneratorAppParams
        ) -> GeneratedTexts:
            document = Document(
                body=text_generator_params.body,
                meta=text_generator_params.meta
            )
            inp_encoding = self._tokenizer.encode_document(
                document=document,
                with_eos=False
            )
            out_encodings = self._generator(
                encoding=inp_encoding,
                params=text_generator_params
            )
            texts = [self._tokenizer.decode_encoding(e) for e in out_encodings]
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
