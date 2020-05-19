from full_stack_transformer.tasks.common.tokenizers.ru_transformers import (
    RuTransformersTokenizerMixin,
    get_base_tokenizer,
    preprocess
)
from full_stack_transformer.tasks.document_decoder import DocumentTokenizer
from full_stack_transformer.tasks.document_decoder.text_input import DocumentInput


class RuTransformersDocumentTokenizer(
    RuTransformersTokenizerMixin,
    DocumentTokenizer
):
    def __init__(
            self,
            max_meta_len: int,
            max_body_len: int,
            ignore_meta_prob: float
    ):
        super().__init__(
            tokenizer=get_base_tokenizer(),
            max_meta_len=max_meta_len,
            max_body_len=max_body_len,
            ignore_meta_prob=ignore_meta_prob,
            pad_token='<pad>'
        )

    def _preprocess_input(self, text_input: DocumentInput) -> DocumentInput:
        body = preprocess(text_input.body)
        meta = preprocess(text_input.meta)

        new_input = DocumentInput(body=body, meta=meta)

        return new_input
