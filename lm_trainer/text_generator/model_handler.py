from typing import Tuple

import torch
import torch.nn as nn
import transformers


class ModelHandlerError(Exception):
    pass


class ModelHandler(nn.Module):
    @property
    def device(self):
        return self._model.parameters().__next__().device

    def __init__(self, model: transformers.GPT2LMHeadModel):
        super().__init__()
        self._model = model
        _check_model_validity(self._model)

    def forward(self, input_ids, past) -> Tuple[torch.tensor, torch.tensor]:
        _check_inputs_validity(input_ids, past)

        self._model.eval()
        with torch.no_grad():
            input_ids, past = self._to_device(input_ids, past)
            logits, past, _ = self._model(input_ids=input_ids, past=past)

        return logits, past

    def _to_device(self, input_ids, past):
        input_ids = input_ids.to(self._model.device)
        if past is not None:
            past = [p.to(self._model.device) for p in past]

        return input_ids, past


def _check_model_validity(model) -> None:
    if not isinstance(model, transformers.GPT2LMHeadModel):
        raise ModelHandlerError(
            '`ModelHandler` works only with `GPT2LMHeadModel` '
            'model instance.')
    elif not getattr(model, '_do_output_past', None):
        raise ModelHandlerError(
            'Model has no attribute `_do_output_past` or it set to False. '
            'Must be equal to True.')
    elif not model.config.output_past:
        raise ModelHandlerError(
            '`output_past` must be set to True. Check model config.')


def _check_inputs_validity(input_ids, past):
    if past is not None and input_ids.size()[1] != 1:
        raise ModelHandlerError(
            'If `past` is provided, `input_ids`  must have (batch_size, 1) '
            'shape.')
