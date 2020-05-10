from typing import Optional

import torch.nn as nn
import transformers

from full_stack_transformer.language_modelling.data_structures import (
    LanguageModelOutput,
    LanguageModelInput
)
from full_stack_transformer.language_modelling.modelling.losses import \
    unlikelihood_candidates_loss


class LanguageModel(nn.Module):

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

    def forward(self, inp: LanguageModelInput) -> LanguageModelOutput:
        lm_loss, logits, past, hidden = self.lm_head_model(
            input_ids=inp.input_ids,
            token_type_ids=inp.token_type_ids,
            labels=inp.lm_labels,
            past=inp.past
        )

        if self._ul_alpha is not None:
            ul_loss = unlikelihood_candidates_loss(
                logits=logits,
                target=inp.input_ids
            )

            loss = lm_loss + self._ul_alpha * ul_loss
        else:
            loss = lm_loss
            ul_loss = None

        output = LanguageModelOutput(
            lm_loss=lm_loss,
            ul_loss=ul_loss,
            loss=loss,
            logits=logits,
            past=past,
            hidden=hidden
        )

        return output

    def infer(self, inp: LanguageModelInput) -> LanguageModelOutput:
        """Performs forward pass without loss calculation."""

        logits, past, hidden = self.lm_head_model(
            input_ids=inp.input_ids,
            token_type_ids=inp.token_type_ids,
            past=inp.past
        )

        output = LanguageModelOutput(
            logits=logits,
            past=past,
            hidden=hidden
        )

        return output
