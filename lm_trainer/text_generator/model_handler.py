from typing import Tuple

import torch
import torch.nn as nn
import transformers

# lm_prediction_scores, mc_prediction_scores, past, hidden_states
ModelOutputT = Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]


class ModelHandlerError(Exception):
    pass


class ModelHandler(nn.Module):
    @property
    def device(self):
        return self._model.parameters().__next__().device

    def __init__(self, model: transformers.GPT2DoubleHeadsModel):
        super().__init__()
        self._model = model
        _check_model_validity(self._model)

    def forward(self, input_ids, token_type_ids, past) -> ModelOutputT:
        _check_inputs_validity(input_ids, token_type_ids, past)

        model_output = self._model(
            input_ids=input_ids, token_type_ids=token_type_ids, past=past)

        return model_output


def _check_model_validity(model) -> None:
    if not isinstance(model, transformers.GPT2DoubleHeadsModel):
        raise ModelHandlerError(
            '`ModelHandler` works only with `GPT2DoubleHeadsModel` '
            'model instance.')
    elif not getattr(model, '_do_output_past', None):
        raise ModelHandlerError(
            'Model has no attribute `_do_output_past` or it set to False. '
            'Must be equal to True.')
    elif not model.config.output_past:
        raise ModelHandlerError(
            '`output_past` must be set to True. Check model config.')
    elif not model.config.output_hidden_states:
        raise ModelHandlerError(
            '`output_hidden_states` must be set to True. '
            'Check model config.')
    elif model.config.num_labels != 2:
        raise ModelHandlerError(
            '`num_labels` in model config must be equal to 2.')


def _check_inputs_validity(input_ids, token_type_ids, past):
    if input_ids.size() != token_type_ids.size():
        raise ModelHandlerError(
            '`input_ids` and `token_type_ids` must have the same shape.')
    elif past is not None and input_ids.size()[1] != 1:
        raise ModelHandlerError(
            'If `past` is provided, `input_ids` and `token_type_ids` must have '
            '(batch_size, 1) shape.')
