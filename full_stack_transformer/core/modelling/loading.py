import copy
import pathlib
from typing import Optional, Union

import transformers


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


def _modify_transformers_config(
        config: transformers.PretrainedConfig
) -> transformers.PretrainedConfig:
    config_copy = copy.deepcopy(config)
    config_copy.output_past = True
    config_copy.output_hidden_states = True
    return config_copy


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
