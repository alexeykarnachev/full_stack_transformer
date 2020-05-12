import pathlib
import re
from typing import Optional

from tokenizers import SentencePieceBPETokenizer

from full_stack_transformer.tasks.document_lm.text_input import DocumentInput
from full_stack_transformer.tasks.document_lm.tokenizer import \
    DocumentTokenizer

_THIS_DIR = pathlib.Path(__file__).parent
_VOCAB = _THIS_DIR / '..' / 'static' / 'ru_transformers_sp' / 'vocab.json'
_MERGES = _THIS_DIR / '..' / 'static' / 'ru_transformers_sp' / 'merges.txt'

_NEW_LINE_REP = '<|n|>'


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
            max_meta_len=max_meta_len,
            max_body_len=max_body_len,
            ignore_meta_prob=ignore_meta_prob,
            pad_token='<pad>'
        )

    def _preprocess_input(self, text_input: DocumentInput) -> DocumentInput:
        body = _preprocess(text_input.body)
        meta = _preprocess(text_input.meta)

        new_input = DocumentInput(body=body, meta=meta)

        return new_input

    def _postprocess_text(self, text: str) -> str:
        return _postrpocess(text=text)


def _preprocess(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None

    if text and text[0] != ' ':
        text = ' ' + text

    text = re.sub(r'(?=[^ ])([\W])([\w])', r'\g<1> \g<2>', text)
    text = text.replace('\n', f' {_NEW_LINE_REP}')
    return text


def _postrpocess(text: str) -> str:
    text = re.sub(re.escape(_NEW_LINE_REP), '\n', text)
    text = re.sub(r'([\n(]) (\w)', r'\g<1>\g<2>', text)
    text = re.sub(r'(\W|^)([«"''\n(]|^) (\w)', r'\g<1>\g<2>\g<3>', text)
    text = re.sub(r'(\w)- (\w)', r'\g<1>-\g<2>', text)
    return text
