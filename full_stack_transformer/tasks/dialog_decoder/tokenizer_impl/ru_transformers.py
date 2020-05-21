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
        orig_data = text_input.__dict__
        orig_data['tags'] = preprocess(orig_data['tags'])
        orig_data['persona'] = preprocess(orig_data['persona'])
        orig_data['utterances'] = [preprocess(ut) for ut in orig_data['utterances']]
        new_input = DialogInput(**orig_data)

        return new_input
