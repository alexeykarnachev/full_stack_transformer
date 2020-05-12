import pathlib

from tokenizers import ByteLevelBPETokenizer

from full_stack_transformer.tasks.document_lm.text_input import DocumentInput
from full_stack_transformer.tasks.document_lm.tokenizer import \
    DocumentTokenizer

_THIS_DIR = pathlib.Path(__file__).parent
_VOCAB = _THIS_DIR / '..' / 'static' / 'gpt2_bpe' / 'vocab.json'
_MERGES = _THIS_DIR / '..' / 'static' / 'gpt2_bpe' / 'merges.txt'


class HFGPT2DocumentTokenizer(DocumentTokenizer):
    def __init__(
            self,
            max_meta_len: int,
            max_body_len: int,
            ignore_meta_prob: float
    ):
        tokenizer = ByteLevelBPETokenizer(
            vocab_file=str(_VOCAB),
            merges_file=str(_MERGES)
        )

        super().__init__(
            tokenizer=tokenizer,
            pad_token='<pad>',
            max_meta_len=max_meta_len,
            max_body_len=max_body_len,
            ignore_meta_prob=ignore_meta_prob
        )

    def _preprocess_input(self, text_input: DocumentInput) -> DocumentInput:
        return text_input

    def _postprocess_text(self, text: str) -> str:
        return text
