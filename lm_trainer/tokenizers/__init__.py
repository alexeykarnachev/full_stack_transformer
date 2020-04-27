import tokenizers

from lm_trainer.tokenizers.ru_transformers_tokenizer import (
    RuTransformersTokenizer
)
from lm_trainer.utilities import factory


def get_tokenizer(
        tokenizer_cls_name: str
) -> tokenizers.implementations.BaseTokenizer:
    tokenizer_cls_path = f'lm_trainer.tokenizers.{tokenizer_cls_name}'
    tokenizer = factory.get_object(tokenizer_cls_path)
    return tokenizer
