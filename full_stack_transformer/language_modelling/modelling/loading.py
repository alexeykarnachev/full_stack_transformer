import copy
import json
import pathlib
import re
from typing import Optional, Union, Mapping

import torch
import transformers

from full_stack_transformer.language_modelling.modelling.model import LanguageModel
from full_stack_transformer.language_modelling.tokenization.tokenizer import get_tokenizer, DocumentTokenizer


class ModelLoadingError(Exception):
    pass


def load_transformer_model_from_path(
        model_path: Union[str, pathlib.Path],
        vocab_size: Optional[int]
) -> transformers.PreTrainedModel:
    config = transformers.AutoConfig.from_pretrained(model_path)
    modified_config = _modify_transformers_config(config)

    model = transformers.AutoModelForPreTraining.from_pretrained(
        pretrained_model_name_or_path=model_path,
        config=modified_config
    )

    _resize_embeddings_if_needed(model, vocab_size)

    return model


def initialize_transformer_model_from_config(
        config: transformers.PretrainedConfig,
        vocab_size: Optional[int]
) -> transformers.PreTrainedModel:
    modified_config = _modify_transformers_config(config)
    model = transformers.AutoModelForPreTraining.from_config(modified_config)

    _resize_embeddings_if_needed(model, vocab_size)

    return model


def load_language_model_from_checkpoint(
        ckpt: Mapping,
        device: Union[torch.device, str],
        unlikelihood_alpha: Optional[float]
) -> LanguageModel:
    transformer = _load_transformer_from_checkpoint(ckpt=ckpt, device=device)
    model = LanguageModel(transformer, unlikelihood_alpha=unlikelihood_alpha)
    return model


def _load_transformer_from_checkpoint(
        ckpt: Mapping,
        device: Union[torch.device, str]
):
    model_config = _load_model_config_from_ckpt(ckpt)

    # Here we need to load tokenizer just for vocabulary size.
    tokenizer = load_tokenizer_from_checkpoint(
        ckpt, ignore_meta_prob=0, max_meta_len=0, max_body_len=0
    )

    model = initialize_transformer_model_from_config(
        config=model_config,
        vocab_size=tokenizer.vocab_size
    )

    model_state_dict = _load_state_dict_from_ckpt(ckpt)
    model.load_state_dict(model_state_dict)
    model = model.to(device)

    return model


def load_tokenizer_from_checkpoint(
        ckpt: Mapping,
        max_body_len: int,
        max_meta_len: int,
        ignore_meta_prob: float = 0.0
) -> DocumentTokenizer:
    name = ckpt['hparams']['tokenizer_class_name']
    tokenizer = get_tokenizer(
        name=name,
        max_meta_len=max_meta_len,
        max_body_len=max_body_len,
        ignore_meta_prob=ignore_meta_prob
    )
    return tokenizer


def _load_model_config_from_ckpt(ckpt):
    model_config = ckpt['hparams']['transformer_config']
    model_config = json.loads(model_config)
    model_config = transformers.GPT2Config(**model_config)

    return model_config


def _modify_transformers_config(
        config: transformers.PretrainedConfig
) -> transformers.PretrainedConfig:
    config_copy = copy.deepcopy(config)
    config_copy.output_past = True
    config_copy.output_hidden_states = True
    return config_copy


def _load_state_dict_from_ckpt(ckpt):
    pl_state_dict = ckpt['state_dict']
    model_state_dict = {}

    for k, v in pl_state_dict.items():
        matched_key = re.search(r'_model\.lm_head_model\.(.+)', k)
        if matched_key:
            key = matched_key.group(1)
            model_state_dict[key] = v

    return model_state_dict


def _resize_embeddings_if_needed(
        model: transformers.PreTrainedModel,
        vocab_size: int
) -> None:
    if vocab_size is not None:
        mean_emb = model.base_model.wte.weight.data.mean(0)
        old_size = model.base_model.wte.weight.data.size()[0]
        n_new = vocab_size - old_size

        if n_new < 0:
            raise ModelLoadingError(
                "Can't resize embeddings: new vocab size can not be less than "
                "the old embeddings number (old vocab size)."
            )

        model.resize_token_embeddings(vocab_size)
        idx = vocab_size - n_new
        model.base_model.wte.weight.data[idx:] = mean_emb.unsqueeze(0)
