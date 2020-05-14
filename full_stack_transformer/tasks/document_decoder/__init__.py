from full_stack_transformer.tasks.common.tokenizers import get_tokenizer
from full_stack_transformer.tasks.common.tokenizers.document import DocumentTokenizer


def load_tokenizer_from_checkpoint(ckpt) -> DocumentTokenizer:
    config = ckpt['hparams']['tokenizer_config']
    config['ignore_meta_prob'] = 0
    tokenizer = get_tokenizer(**config)
    return tokenizer