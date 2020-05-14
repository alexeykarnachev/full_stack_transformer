from typing import List, Optional, Sequence, Tuple

import more_itertools
from tokenizers.implementations import BaseTokenizer

from full_stack_transformer.core.constants import LOSS_IGNORE
from full_stack_transformer.core.encoding import Encoding
from full_stack_transformer.core.tokenizer import Tokenizer
from full_stack_transformer.tasks.common.text_inputs.dialog import DialogInput

_PERSONA_SPEAKER = '[PERSONA_SPEAKER]'
_NOT_PERSONA_SPEAKER = '[NOT_PERSONA_SPEAKER]'
_END_OF_PERSONA = '[END_OF_PERSONA]'
_END_OF_TAGS = '[END_OF_TAGS]'

_SPECIAL_TOKENS = [
    _PERSONA_SPEAKER,
    _NOT_PERSONA_SPEAKER,
    _END_OF_PERSONA,
    _END_OF_TAGS
]


class DialogTokenizer(Tokenizer):
    def __init__(
            self,
            tokenizer: BaseTokenizer,
            max_tags_len: int,
            max_pers_len: int,
            max_dialog_len: int,
            **kwargs
    ):
        super().__init__(tokenizer, **kwargs)

        self._max_tags_len = max_tags_len
        self._max_pers_len = max_pers_len
        self._max_dialog_len = max_dialog_len
        self.add_special_tokens({'additional_special_tokens': _SPECIAL_TOKENS})

    def _encode_for_train(
            self,
            text_input: DialogInput
    ) -> List[Encoding]:
        return self._encode(text_input=text_input, train=True)

    def _encode_for_inference(
            self,
            text_input: DialogInput
    ) -> List[Encoding]:
        return self._encode(text_input=text_input, train=False)

    def _encode(self, text_input: DialogInput, train: bool) -> List[Encoding]:

        tag_ids, tag_types, tag_labels = self._encode_meta(
            string=text_input.tags,
            end_token=_END_OF_TAGS,
            max_len=self._max_tags_len
        )

        encodings = []

        if text_input.persona_0:
            pers_ids, pers_types, pers_labels = self._encode_meta(
                string=text_input.persona_0,
                end_token=_END_OF_PERSONA,
                max_len=self._max_pers_len
            )

            dlg_ids, dlg_types, dlg_labels = self._encode_dialog(
                utterances=text_input.utterances,
                pers_idx=0,
                train=train
            )

            enc = Encoding(
                token_ids=tag_ids + pers_ids + dlg_ids,
                lm_labels=tag_labels + pers_labels + dlg_labels,
                token_type_ids=tag_types + pers_types + dlg_types
            )
            encodings.append(enc)

        if text_input.persona_1:
            pers_ids, pers_types, pers_labels = self._encode_meta(
                string=text_input.persona_1,
                end_token=_END_OF_PERSONA,
                max_len=self._max_pers_len
            )

            dlg_ids, dlg_types, dlg_labels = self._encode_dialog(
                utterances=text_input.utterances,
                pers_idx=1,
                train=train
            )

            enc = Encoding(
                token_ids=tag_ids + pers_ids + dlg_ids,
                lm_labels=tag_labels + pers_labels + dlg_labels,
                token_type_ids=tag_types + pers_types + dlg_types
            )
            encodings.append(enc)

        if not text_input.persona_0 and not text_input.persona_1:
            pers_ids, pers_types, pers_labels = self._encode_meta(
                string=None,
                end_token=_END_OF_PERSONA,
                max_len=self._max_pers_len
            )

            dlg_ids, dlg_types, dlg_labels = self._encode_dialog(
                utterances=text_input.utterances,
                pers_idx=None,
                train=train
            )

            enc = Encoding(
                token_ids=tag_ids + pers_ids + dlg_ids,
                lm_labels=tag_labels + pers_labels + dlg_labels,
                token_type_ids=tag_types + pers_types + dlg_types
            )
            encodings.append(enc)

        return encodings

    def _encode_meta(
            self,
            string: Optional[str],
            end_token: str,
            max_len: int
    ) -> Tuple[List[int], List[int], List[int]]:
        string = string or ''
        string = f'{string}{end_token}'
        ids = self.encode(string)
        types = [ids[-1]] * len(ids)
        labels = [LOSS_IGNORE] * len(ids)
        return ids[-max_len:], types[-max_len:], labels[-max_len:]

    def _encode_dialog(
            self,
            utterances: Sequence[str],
            pers_idx: Optional[int],
            train: bool
    ) -> Tuple[List[int], List[int], List[int]]:
        utts = []

        ignore_loss = []
        for idx, ut in enumerate(utterances):
            if pers_idx is None or idx % 2 == pers_idx:
                pers_tok, not_pers_tok = _PERSONA_SPEAKER, _NOT_PERSONA_SPEAKER
                ignore_loss.append(False)
            else:
                pers_tok, not_pers_tok = _NOT_PERSONA_SPEAKER, _PERSONA_SPEAKER
                ignore_loss.append(True)

            ut = f'{pers_tok}{ut}{self.eos_token}'
            if idx == len(utterances) - 1 and not train:
                ut = f'{ut}{not_pers_tok}'

            utts.append(ut)

        utts_ids = self.batch_encode_plus(utts, add_special_tokens=False)
        utts_ids = list(utts_ids['input_ids'])
        tok_types = [[u[0]] * len(u) for u in utts_ids]

        labels = []
        for ids, ignore in zip(utts_ids, ignore_loss):
            if ignore:
                labs = [LOSS_IGNORE] * len(ids)
            else:
                labs = list(ids)
                labs[0] = LOSS_IGNORE

            labels.append(labs)

        ids = list(more_itertools.flatten(utts_ids))[-self._max_dialog_len:]
        types = list(more_itertools.flatten(tok_types))[-self._max_dialog_len:]
        labels = list(more_itertools.flatten(labels))[-self._max_dialog_len:]

        return ids, types, labels


def _not_ignore_utt(utt_id: int, pers_id: Optional[int]):
    if pers_id == 0:
        return abs((utt_id % 2) - 1)
    elif pers_id == 1:
        return utt_id % 2
    elif pers_id is None:
        return True
    else:
        raise ValueError(f'Bad person id: {pers_id}')
