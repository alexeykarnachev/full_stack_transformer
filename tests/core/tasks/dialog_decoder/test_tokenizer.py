from full_stack_transformer.tasks.dialog_decoder.text_input import DialogInput
from full_stack_transformer.tasks.dialog_decoder.tokenizer import (
    _PERSONA_SPEAKER,
    _NOT_PERSONA_SPEAKER,
    _END_OF_PERSONA,
    _END_OF_TAGS
)
from full_stack_transformer.tasks.dialog_decoder.tokenizer_impl.ru_transformers import \
    RuTransformersDialogTokenizer


def test_tokenizer():
    tokenizer = RuTransformersDialogTokenizer(
        max_tags_len=100,
        max_pers_len=100,
        max_dialog_len=100
    )

    utterances = ['ААПП', 'ББББ', 'АВВА']
    persona = 'П'
    tags = 'T'
    persona_idx = 0

    text_input = DialogInput(
        utterances=utterances,
        persona_idx=persona_idx,
        persona=persona,
        tags=tags
    )

    eos = tokenizer.eos_token
    expected_decoded_ids = f'{tags}{_END_OF_TAGS} {persona}{_END_OF_PERSONA}'
    not_pers_tok = _NOT_PERSONA_SPEAKER
    for i_utt, utt in enumerate(utterances):
        if (i_utt % 2) == persona_idx:
            pers_tok, not_pers_tok = _PERSONA_SPEAKER, _NOT_PERSONA_SPEAKER
        else:
            pers_tok, not_pers_tok = _NOT_PERSONA_SPEAKER, _PERSONA_SPEAKER

        expected_decoded_ids += f'{pers_tok} {utt}{eos}'

    encoding = tokenizer.encode_for_train(text_input=text_input)
    decoded_ids = tokenizer.decode(encoding.token_ids)
    assert decoded_ids == expected_decoded_ids

    expected_decoded_ids += not_pers_tok
    encoding = tokenizer.encode_for_inference(text_input=text_input)
    decoded_ids = tokenizer.decode(encoding.token_ids)
    assert decoded_ids == expected_decoded_ids
