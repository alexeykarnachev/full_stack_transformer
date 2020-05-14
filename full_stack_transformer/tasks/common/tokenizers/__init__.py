from full_stack_transformer.core.tokenizer import Tokenizer

from full_stack_transformer.utilities.factory import get_object


def get_tokenizer(name: str, **kwargs):
    path = f'full_stack_transformer.tasks.common.{name}'
    obj = get_object(path, **kwargs)

    if not isinstance(obj, Tokenizer):
        raise ValueError(f'{path} is not a Tokenizer.')

    return obj
