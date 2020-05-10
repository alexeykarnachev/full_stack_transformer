import pathlib
import re

from tokenizers import SentencePieceBPETokenizer

from full_stack_transformer.language_modelling.tokenization.tokenizer import DocumentTokenizer

_THIS_DIR = pathlib.Path(__file__).parent
_VOCAB = _THIS_DIR / 'data' / 'ru_transformers_tokenizer' / 'vocab.json'
_MERGES = _THIS_DIR / 'data' / 'ru_transformers_tokenizer' / 'merges.txt'


class RuTransformersDocumentTokenizer(DocumentTokenizer):
    def __init__(
            self,
            max_meta_len: int,
            max_body_len: int,
            ignore_meta_prob: float
    ):
        tokenizer = SentencePieceBPETokenizer(
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
        if text and text[0] != ' ':
            text = ' ' + text

        text = re.sub(r'(?=[^ ])([\W])([\w])', r'\g<1> \g<2>', text)
        text = text.replace('\n', ' <|n|>')
        return text
