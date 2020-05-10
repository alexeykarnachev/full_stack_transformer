import copy
import pathlib
from typing import Optional, Union

import transformers


def load_transformer_model_from_path(
        model_path: Union[str, pathlib.Path],
        vocab_size: Optional[int]) -> transformers.PreTrainedModel:
    config = transformers.AutoConfig.from_pretrained(model_path)
    modified_config = _modify_transformers_config(config)

    model = transformers.AutoModelForPreTraining.from_pretrained(
        pretrained_model_name_or_path=model_path, config=modified_config)

    _resize_embeddings_if_needed(model, vocab_size)

    return model


def _modify_transformers_config(
        config: transformers.PretrainedConfig) -> transformers.PretrainedConfig:
    config_copy = copy.deepcopy(config)
    config_copy.output_past = True
    config_copy.output_hidden_states = True
    return config_copy


def _resize_embeddings_if_needed(
        model: transformers.PreTrainedModel,
        vocab_size: int) -> None:
    if vocab_size is not None:
        model.resize_token_embeddings(vocab_size)
