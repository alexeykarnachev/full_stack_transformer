from full_stack_transformer.tasks.common.tokenizers.hf_gpt2 import (
    HFGPT2TokenizerMixin,
    get_base_tokenizer
)
from full_stack_transformer.tasks.document_decoder.tokenizer import \
    DocumentTokenizer


class HFGPT2DocumentTokenizer(HFGPT2TokenizerMixin, DocumentTokenizer):
    def __init__(
            self,
            max_meta_len: int,
            max_body_len: int,
            ignore_meta_prob: float = 0,
    ):
        super().__init__(
            tokenizer=get_base_tokenizer(),
            pad_token='<pad>',
            max_meta_len=max_meta_len,
            max_body_len=max_body_len,
            ignore_meta_prob=ignore_meta_prob
        )
