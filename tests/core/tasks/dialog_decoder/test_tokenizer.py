import pytest

from full_stack_transformer.tasks.dialog_decoder.text_input import DialogInput
from full_stack_transformer.tasks.dialog_decoder.tokenizer_impl.ru_transformers import \
    RuTransformersDialogTokenizer


@pytest.mark.parametrize(
    'max_tags_len,max_pers_len,max_dialog_len', [
        (10, 10, 10)
    ]
)
def test_tokenizer(max_tags_len, max_pers_len, max_dialog_len):
    tokenizer = RuTransformersDialogTokenizer(
        max_tags_len=max_tags_len,
        max_pers_len=max_pers_len,
        max_dialog_len=max_dialog_len
    )

    text_input = DialogInput(
        utterances=['Мама мыла раму', 'Где мыла?'],
        persona_idx=0,
        persona='',
        tags=''
    )

    encoding = tokenizer.encode_for_inference(text_input=text_input)
    print(tokenizer.decode(encoding.token_type_ids))