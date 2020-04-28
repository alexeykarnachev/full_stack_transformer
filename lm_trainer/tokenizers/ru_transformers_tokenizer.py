import pathlib
import re

import tokenizers

from lm_trainer.tokenizers.tokenizer import Tokenizer

_THIS_DIR = pathlib.Path(__file__).parent
_VOCAB = _THIS_DIR / 'data' / 'ru_transformers_yt' / 'vocab.json'
_MERGES = _THIS_DIR / 'data' / 'ru_transformers_yt' / 'merges.txt'

_NEW_LINE_REP = '<|n|>'
_NEW_LINE_PAT = re.compile(r'\n')

_PREPROC_PAT = re.compile(r'(?=[^ ])([\W])([\w])')
_PREPROC_REP = r'\g<1> \g<2>'
_DOC_START = '[DOC_START]'
_DOC_END = '[DOC_END]'


class RuTransformersTokenizer(Tokenizer):
    """BPE Tokenizer class for model from
    https://github.com/mgrankin/ru_transformers.
    """

    def get_pad_token(self) -> str:
        return '<pad>'

    def get_eos_token(self) -> str:
        return _DOC_END

    def get_bos_token(self) -> str:
        return _DOC_START

    def __init__(self):
        super().__init__(vocab_file=str(_VOCAB), merges_file=str(_MERGES))
        self._add_tokens()

    def _add_tokens(self):
        new_line_sep_token = tokenizers.AddedToken(_NEW_LINE_REP)
        bos_token = tokenizers.AddedToken(_DOC_START)
        eos_token = tokenizers.AddedToken(_DOC_END)

        self.add_tokens([new_line_sep_token, bos_token, eos_token])

    def preprocess(self, sequence) -> str:
        """Text preprocessor for ru_transformers tokenizer.
        https://github.com/mgrankin/ru_transformers/blob/master/yt_encoder.py

        Args:
            sequence (str): Input string.

        Returns:
            Processed string.
        """
        if sequence and sequence[0] != ' ':
            sequence = ' ' + sequence

        sequence = _PREPROC_PAT.sub(_PREPROC_REP, sequence)
        sequence = _NEW_LINE_PAT.sub(_NEW_LINE_REP, sequence)

        return sequence

    def postprocess(self, sequence) -> str:
        """Performs postprocessing on the detokenized sequence."""
        sequence = re.sub(re.escape(_NEW_LINE_REP), '\n', sequence)
        sequence = re.sub(r'( )?(<\|n\|>)( )?', r'\n', sequence)
        sequence = re.sub(r'([\n(]) (\w)', r'\g<1>\g<2>', sequence)
        sequence = re.sub(r'(\W|^)([«"''\n(]|^) (\w)', r'\g<1>\g<2>\g<3>', sequence)
        sequence = re.sub(r'(\w)- (\w)', r'\g<1>-\g<2>', sequence)
        return sequence
