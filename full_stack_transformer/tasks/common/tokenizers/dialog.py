from typing import List, Optional, Sequence

from tokenizers.implementations import BaseTokenizer

from full_stack_transformer.core.constants import LOSS_IGNORE
from full_stack_transformer.core.encoding import Encoding
from full_stack_transformer.core.tokenizer import Tokenizer
from full_stack_transformer.tasks.common.text_inputs.dialog import DialogInput

_START_OF_UTTERANCE = '[START_OF_UTTERANCE]'
_END_OF_PERSONA = '[END_OF_PERSONA]'
_END_OF_TAGS = '[END_OF_TAGS]'

_SPECIAL_TOKENS = [_START_OF_UTTERANCE, _END_OF_PERSONA, _END_OF_TAGS]


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

    def _encode_for_train(self, text_input: DialogInput) -> List[Encoding]:
        utts_tok_ids = self._get_utts_tok_ids(utts=text_input.utterances)
        tags_tok_ids = self._get_tags_tok_ids(tags=text_input.tags)

        encs = []

        perss = [text_input.persona_0, text_input.persona_1]

        if not perss[0] and not perss[1]:
            enc = self._get_pers_enc(
                pers=None,
                pers_id=None,
                tags_tok_ids=tags_tok_ids,
                utts_tok_ids=utts_tok_ids
            )
            encs.append(enc)
        else:
            for i_pers, pers in enumerate(perss):
                if pers is not None:
                    enc = self._get_pers_enc(
                        pers=pers,
                        pers_id=i_pers,
                        tags_tok_ids=tags_tok_ids,
                        utts_tok_ids=utts_tok_ids
                    )
                    encs.append(enc)

        return encs

    def _get_pers_enc(
            self,
            pers: Optional[str],
            pers_id: Optional[int],
            tags_tok_ids: List[int],
            utts_tok_ids: Sequence[List[int]]
    ) -> Encoding:
        pers_tok_ids = self._get_pers_tok_ids(pers=pers)
        tok_ids = tags_tok_ids + pers_tok_ids
        tags_labels = [LOSS_IGNORE] * len(tags_tok_ids)
        pers_labels = [LOSS_IGNORE] * len(pers_tok_ids)
        labels = tags_labels + pers_labels

        dialog_tok_ids = []
        dialog_labels = []

        for i_utt, utt_tok_ids in enumerate(utts_tok_ids):
            if _not_ignore_utt(utt_id=i_utt, pers_id=pers_id):
                utt_labels = list(utt_tok_ids)
                utt_labels[0] = LOSS_IGNORE
            else:
                utt_labels = [LOSS_IGNORE] * len(utt_tok_ids)

            dialog_labels += utt_labels
            dialog_tok_ids += utt_tok_ids

        dialog_tok_ids = dialog_tok_ids[-self._max_dialog_len:]
        dialog_labels = dialog_labels[-self._max_dialog_len:]

        tok_ids += dialog_tok_ids
        labels += dialog_labels

        enc = Encoding(token_ids=tok_ids, lm_labels=labels)
        return enc

    def _get_utts_tok_ids(self, utts: Sequence[str]) -> List[List[int]]:
        tok_ids = []

        for ut in utts:
            ut = f'{_START_OF_UTTERANCE}{ut}{self.eos_token}'
            tok_ids.append(self.encode(ut))

        return tok_ids

    def _get_tags_tok_ids(self, tags: Optional[str]) -> List[int]:
        tags = tags or ''
        tags = f'{tags}{_END_OF_TAGS}'
        tok_ids = self.encode(tags)[-self._max_tags_len:]

        return tok_ids

    def _get_pers_tok_ids(self, pers: Optional[str]) -> List[int]:
        pers = pers or ''
        pers = f'{pers}{_END_OF_PERSONA}'
        tok_ids = self.encode(pers)[-self._max_pers_len:]

        return tok_ids

    def _encode_for_inference(
            self,
            text_input: DialogInput
    ) -> List[Encoding]:
        pass


def _not_ignore_utt(utt_id: int, pers_id: Optional[int]):
    if pers_id == 0:
        return abs((utt_id % 2) - 1)
    elif pers_id == 1:
        return utt_id % 2
    elif pers_id is None:
        return True
    else:
        raise ValueError(f'Bad person id: {pers_id}')
