import pathlib

from tokenizers import ByteLevelBPETokenizer

from full_stack_transformer.tasks.common.text_inputs.document import DocumentInput
from full_stack_transformer.tasks.common.tokenizers.dialog import DialogTokenizer
from full_stack_transformer.tasks.common.tokenizers.document import DocumentTokenizer

_THIS_DIR = pathlib.Path(__file__).parent
_VOCAB = _THIS_DIR / 'static' / 'gpt2_bpe' / 'vocab.json'
_MERGES = _THIS_DIR / 'static' / 'gpt2_bpe' / 'merges.txt'


class HFGPT2TokenizerMixin:
    def _preprocess_input(self, text_input: DocumentInput) -> DocumentInput:
        return text_input

    def _postprocess_text(self, text: str) -> str:
        return text


class HFGPT2DocumentTokenizer(HFGPT2TokenizerMixin, DocumentTokenizer):
    def __init__(
            self,
            max_meta_len: int,
            max_body_len: int,
            ignore_meta_prob: float = 0,
    ):
        super().__init__(
            tokenizer=_get_base_tokenizer(),
            pad_token='<pad>',
            max_meta_len=max_meta_len,
            max_body_len=max_body_len,
            ignore_meta_prob=ignore_meta_prob
        )


class HFGPT2DialogTokenizer(HFGPT2TokenizerMixin, DialogTokenizer):
    def __init__(
            self,
            max_tags_len: int,
            max_pers_len: int,
            max_dialog_len: int,
    ):
        super().__init__(
            tokenizer=_get_base_tokenizer(),
            pad_token='<pad>',
            max_tags_len=max_tags_len,
            max_pers_len=max_pers_len,
            max_dialog_len=max_dialog_len
        )


def _get_base_tokenizer():
    tokenizer = ByteLevelBPETokenizer(
        vocab_file=str(_VOCAB),
        merges_file=str(_MERGES)
    )

    return tokenizer
