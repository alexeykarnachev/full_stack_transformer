import pathlib
import re

import tokenizers

from full_stack_transformer.tokenization import Tokenizer

_THIS_DIR = pathlib.Path(__file__).parent
_VOCAB = _THIS_DIR / 'data' / 'gpt2_bpe' / 'vocab.json'
_MERGES = _THIS_DIR / 'data' / 'gpt2_bpe' / 'merges.txt'

_DOC_START = '[DOC_START]'
_DOC_END = '[DOC_END]'
_PAD = '[PAD]'


class GPT2Tokenizer(Tokenizer):
    def get_pad_token(self) -> str:
        return _PAD

    def get_eos_token(self) -> str:
        return _DOC_END

    def get_bos_token(self) -> str:
        return _DOC_START

    def __init__(self):
        tokenizer = tokenizers.implementations.ByteLevelBPETokenizer(
            vocab_file=str(_VOCAB), merges_file=str(_MERGES))
        super().__init__(tokenizer=tokenizer)
        self._add_tokens()

    def _add_tokens(self):
        bos_token = tokenizers.AddedToken(_DOC_START)
        eos_token = tokenizers.AddedToken(_DOC_END)
        pad_token = tokenizers.AddedToken(_PAD)

        self.add_tokens([bos_token, eos_token, pad_token])

    def _preprocess(self, sequence) -> str:
        return sequence

    def _postprocess(self, sequence) -> str:
        sequence = re.sub(re.escape(_DOC_START), '', sequence)
        sequence = re.sub(re.escape(_DOC_END), '', sequence)
        sequence = sequence.strip()
        return sequence
