import pathlib

from tokenizers import ByteLevelBPETokenizer

from full_stack_transformer.tasks.document_decoder.text_input import DocumentInput

_THIS_DIR = pathlib.Path(__file__).parent
_VOCAB = _THIS_DIR / 'static' / 'gpt2_bpe' / 'vocab.json'
_MERGES = _THIS_DIR / 'static' / 'gpt2_bpe' / 'merges.txt'


class HFGPT2TokenizerMixin:
    def _preprocess_input(self, text_input: DocumentInput) -> DocumentInput:
        return text_input

    def _postprocess_text(self, text: str) -> str:
        return text


def get_base_tokenizer():
    tokenizer = ByteLevelBPETokenizer(
        vocab_file=str(_VOCAB),
        merges_file=str(_MERGES)
    )

    return tokenizer
