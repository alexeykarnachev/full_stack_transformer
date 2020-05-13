import json
import re

from transformers import GPT2Config

from full_stack_transformer.core.modelling.loading import initialize_transformer_model_from_config
from full_stack_transformer.tasks.document_lm.modelling.model import DocumentModel
from full_stack_transformer.tasks.document_lm.tokenizer import get_tokenizer, DocumentTokenizer


def load_model_from_checkpoint(ckpt, device):
    state_dict = dict()

    for k, v in ckpt['state_dict'].items():
        new_key = re.search(r'^model\.(.+)$', k).group(1)
        state_dict[new_key] = v

    vocab_size = state_dict['lm_head_model.transformer.wte.weight'].size()[0]

    transformer_config = json.loads(ckpt['hparams']['transformer_config'])
    transformer_config = GPT2Config(**transformer_config)

    lm_head_model = initialize_transformer_model_from_config(
        config=transformer_config,
        vocab_size=vocab_size
    )

    model = DocumentModel(
        lm_head_model=lm_head_model,
        unlikelihood_alpha=None
    )

    model.load_state_dict(state_dict=state_dict)
    model = model.to(device)

    return model


def load_tokenizer_from_checkpoint(ckpt) -> DocumentTokenizer:
    config = ckpt['hparams']['tokenizer_config']
    config['ignore_meta_prob'] = 0
    tokenizer = get_tokenizer(**config)
    return tokenizer
