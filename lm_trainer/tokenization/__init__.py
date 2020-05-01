from lm_trainer.tokenization.ru_transformers_tokenizer import (
    RuTransformersTokenizer
)
from lm_trainer.tokenization.tokenizer import Tokenizer
from lm_trainer.utilities import factory


def get_tokenizer(tokenizer_cls_name: str) -> Tokenizer:
    tokenizer_cls_path = f'lm_trainer.tokenization.{tokenizer_cls_name}'
    return factory.get_object(tokenizer_cls_path)
