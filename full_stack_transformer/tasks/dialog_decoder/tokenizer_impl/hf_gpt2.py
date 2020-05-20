from full_stack_transformer.tasks.common.tokenizers.hf_gpt2 import (
    HFGPT2TokenizerMixin,
    get_base_tokenizer
)
from full_stack_transformer.tasks.dialog_decoder.tokenizer import DialogTokenizer


class HFGPT2DialogTokenizer(HFGPT2TokenizerMixin, DialogTokenizer):
    def __init__(
            self,
            max_tags_len: int,
            max_pers_len: int,
            max_dialog_len: int,
    ):
        super().__init__(
            tokenizer=get_base_tokenizer(),
            pad_token='<pad>',
            max_tags_len=max_tags_len,
            max_pers_len=max_pers_len,
            max_dialog_len=max_dialog_len
        )