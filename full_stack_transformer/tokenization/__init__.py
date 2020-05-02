from full_stack_transformer.tokenization.ru_transformers_tokenizer import (
    RuTransformersTokenizer
)
from full_stack_transformer.tokenization.tokenizer import Tokenizer
from full_stack_transformer.utilities import factory


def get_tokenizer(tokenizer_cls_name: str) -> Tokenizer:
    cls_path = f'full_stack_transformer.tokenization.{tokenizer_cls_name}'
    return factory.get_object(cls_path)
