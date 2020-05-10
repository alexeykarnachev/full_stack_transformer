import pathlib

from tokenizers import ByteLevelBPETokenizer

from full_stack_transformer.language_modelling.tokenization.tokenizer import \
    DocumentTokenizer

_THIS_DIR = pathlib.Path(__file__).parent
_VOCAB = _THIS_DIR / 'data' / 'gpt2_bpe' / 'vocab.json'
_MERGES = _THIS_DIR / 'data' / 'gpt2_bpe' / 'merges.txt'


class GPT2HuggingFaceDocumentTokenizer(DocumentTokenizer):
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

    def prepare_for_tokenization(self, text: str) -> str:
        return text
