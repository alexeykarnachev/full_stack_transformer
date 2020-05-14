import pathlib
import re
from typing import Optional

from tokenizers import SentencePieceBPETokenizer

from full_stack_transformer.tasks.common.text_inputs.dialog import DialogInput
from full_stack_transformer.tasks.common.text_inputs.document import DocumentInput
from full_stack_transformer.tasks.common.tokenizers.dialog import DialogTokenizer
from full_stack_transformer.tasks.common.tokenizers.document import \
    DocumentTokenizer

_THIS_DIR = pathlib.Path(__file__).parent
_VOCAB = _THIS_DIR / 'static' / 'ru_transformers_sp' / 'vocab.json'
_MERGES = _THIS_DIR / 'static' / 'ru_transformers_sp' / 'merges.txt'

_NEW_LINE_REP = '<|n|>'


class RuTransformersTokenizerMixin:
    def _postprocess_text(self, text: str) -> str:
        return _postrpocess(text=text)


class RuTransformersDocumentTokenizer(
    RuTransformersTokenizerMixin,
    DocumentTokenizer
):
    def __init__(
            self,
            max_meta_len: int,
            max_body_len: int,
            ignore_meta_prob: float
    ):
        super().__init__(
            tokenizer=_get_base_tokenizer(),
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
            tokenizer=_get_base_tokenizer(),
            max_tags_len=max_tags_len,
            max_pers_len=max_pers_len,
            max_dialog_len=max_dialog_len,
            pad_token='<pad>'
        )

    def _preprocess_input(self, text_input: DialogInput) -> DialogInput:
        new_input = DialogInput(
            tags=_preprocess(text_input.tags),
            persona_0=_preprocess(text_input.persona_0),
            persona_1=_preprocess(text_input.persona_1),
            utterances=[_preprocess(ut) for ut in text_input.utterances]
        )

        return new_input


def _get_base_tokenizer():
    tokenizer = SentencePieceBPETokenizer(
        vocab_file=str(_VOCAB),
        merges_file=str(_MERGES)
    )

    return tokenizer


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
