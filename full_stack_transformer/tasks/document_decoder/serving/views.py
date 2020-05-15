import re
from typing import Mapping

from fastapi import FastAPI

from full_stack_transformer.tasks.common.text_inputs.document import DocumentInput
from full_stack_transformer.tasks.common.language_generator.generator import LanguageGenerator, \
    LanguageGeneratorParams
from full_stack_transformer.tasks.document_decoder.serving.schemas import (
    GeneratedTexts,
    LanguageGeneratorAppParams
)
from full_stack_transformer.tasks.common.tokenizers.document import DocumentTokenizer


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
                app_params: LanguageGeneratorAppParams
        ) -> GeneratedTexts:
            params = LanguageGeneratorParams(
                app_params.max_number_of_generated_tokens,
                num_return_sequences=app_params.num_return_sequences,
                repetition_penalty=app_params.repetition_penalty,
                temperature=app_params.temperature,
                top_p=app_params.top_p,
                top_k=app_params.top_k
            )
            document = DocumentInput(
                body=app_params.body,
                meta=app_params.meta
            )
            inp_encoding = self._tokenizer.encode_for_inference(
                text_input=document
            )
            encodings = self._generator(
                encoding=inp_encoding,
                params=params
            )
            texts = [self._tokenizer.decode_encoding(e) for e in encodings]
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