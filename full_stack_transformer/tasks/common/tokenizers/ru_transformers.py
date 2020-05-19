import pathlib
import re
from typing import Optional

from tokenizers import SentencePieceBPETokenizer

_THIS_DIR = pathlib.Path(__file__).parent
_VOCAB = _THIS_DIR / 'static' / 'ru_transformers_sp' / 'vocab.json'
_MERGES = _THIS_DIR / 'static' / 'ru_transformers_sp' / 'merges.txt'

_NEW_LINE_REP = '<|n|>'


class RuTransformersTokenizerMixin:
    def _postprocess_text(self, text: str) -> str:
        return postrpocess(text=text)


def get_base_tokenizer():
    tokenizer = SentencePieceBPETokenizer(
        vocab_file=str(_VOCAB),
        merges_file=str(_MERGES)
    )

    return tokenizer


def preprocess(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None

    if text and text[0] != ' ':
        text = ' ' + text

    text = re.sub(r'(?=[^ ])([\W])([\w])', r'\g<1> \g<2>', text)
    text = text.replace('\n', f' {_NEW_LINE_REP}')
    return text


def postrpocess(text: str) -> str:
    text = re.sub(re.escape(_NEW_LINE_REP), '\n', text)
    text = re.sub(r'([\n(]) (\w)', r'\g<1>\g<2>', text)
    text = re.sub(r'(\W|^)([«"''\n(]|^) (\w)', r'\g<1>\g<2>\g<3>', text)
    text = re.sub(r'(\w)- (\w)', r'\g<1>-\g<2>', text)
    return text
