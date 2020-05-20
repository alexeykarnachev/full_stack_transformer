from full_stack_transformer.tasks.common.tokenizers.ru_transformers import (
    RuTransformersTokenizerMixin,
    get_base_tokenizer,
    preprocess
)
from full_stack_transformer.tasks.dialog_decoder.text_input import DialogInput
from full_stack_transformer.tasks.dialog_decoder.tokenizer import DialogTokenizer


class RuTransformersDialogTokenizer(
    RuTransformersTokenizerMixin,
    DialogTokenizer
):
    def __init__(
            self,
            max_tags_len: int,
            max_pers_len: int,
            max_dialog_len: int,
    ):
        super().__init__(
            tokenizer=get_base_tokenizer(),
            max_tags_len=max_tags_len,
            max_pers_len=max_pers_len,
            max_dialog_len=max_dialog_len,
            pad_token='<pad>'
        )

    def _preprocess_input(self, text_input: DialogInput) -> DialogInput:
        new_input = DialogInput(
            tags=preprocess(text_input.tags),
            persona=preprocess(text_input.persona),
            utterances=[preprocess(ut) for ut in text_input.utterances]
        )

        return new_input
