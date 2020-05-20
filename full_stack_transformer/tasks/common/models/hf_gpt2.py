import json
import logging
import re
from typing import Optional

import numpy as np
import torch
import transformers
from transformers import GPT2Config

from full_stack_transformer.core.model_input import ModelInput
from full_stack_transformer.core.model_output import LanguageModelOutput
from full_stack_transformer.core.modelling.loading import initialize_transformer_model_from_config
from full_stack_transformer.core.modelling.model import Model
from full_stack_transformer.core.nn.unlikelihood_candidates_loss import \
    unlikelihood_candidates_loss

_LOGGER = logging.getLogger(__name__)


class HFGPT2Model(Model):
    @property
    def device(self):
        return self.parameters().__next__().device

    def __init__(
            self,
            lm_head_model: transformers.GPT2LMHeadModel,
            unlikelihood_alpha: Optional[float]
    ):
        super().__init__()

        self.lm_head_model = lm_head_model
        self._ul_alpha = unlikelihood_alpha

    def freeze_n_layers(self, n: int) -> None:
        if not n:
            return None
        else:
            for name, params in self.named_parameters():
                layer_number = re.search(r'\.h\.(\d+)\.', name)
                if layer_number:
                    layer_number = int(layer_number.group(1))
                    if layer_number < n:
                        params.requires_grad = False
                        _LOGGER.info(f"Layer `{name}` has been freezed.")

    def forward(self, inp: ModelInput) -> LanguageModelOutput:
        if not isinstance(inp, ModelInput):
            inp = ModelInput(*inp)

        lm_loss, logits, past, hidden = self.lm_head_model(
            input_ids=inp.input_ids,
            token_type_ids=inp.token_type_ids,
            labels=inp.lm_labels,
            past=inp.past
        )

        if self._ul_alpha:
            ul_loss = unlikelihood_candidates_loss(
                logits=logits,
                target=inp.input_ids
            )

            loss = lm_loss + self._ul_alpha * ul_loss
        else:
            loss = lm_loss
            ul_loss = torch.tensor(np.float('nan'))

        output = LanguageModelOutput(
            lm_loss=lm_loss,
            ul_loss=ul_loss,
            loss=loss,
            logits=logits,
            past=past,
            hidden=hidden
        )

        return output

    @torch.no_grad()
    def infer(self, inp: ModelInput) -> LanguageModelOutput:
        """Performs forward pass without loss calculation."""

        logits, past, hidden = self.lm_head_model(
            input_ids=inp.input_ids,
            token_type_ids=inp.token_type_ids,
            past=inp.past
        )

        output = LanguageModelOutput(
            logits=logits,
            past=past,
            hidden=hidden,
            loss=None
        )

        return output


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

    model = HFGPT2Model(
        lm_head_model=lm_head_model,
        unlikelihood_alpha=None
    )

    model.load_state_dict(state_dict=state_dict)
    model = model.to(device)

    return model
