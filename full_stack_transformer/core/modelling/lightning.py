import abc
from typing import Mapping, Tuple, Dict, Sequence

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.base import merge_dicts

from full_stack_transformer.core.model_input import ModelInput
from full_stack_transformer.core.model_output import ModelOutput
from full_stack_transformer.core.modelling.model import Model
from full_stack_transformer.utilities.arguments import ArgparserExtender


class PLModule(LightningModule, ArgparserExtender):
    def __init__(self, model: Model):
        super().__init__()

        self.model = model

    def forward(self, model_inp: ModelInput) -> ModelOutput:
        output = self.model(model_inp)
        return output

    def training_step(self, model_inp: ModelInput, batch_idx: int) -> Dict:
        loss, log = self._step(model_inp=model_inp)
        return {'loss': loss, 'log': log}

    def validation_step(self, model_inp: ModelInput, batch_idx: int) -> Dict:
        loss, log = self._step(model_inp=model_inp)
        return {'val_loss': loss, 'log': log}

    def validation_epoch_end(self, val_step_results: Sequence):
        validation_epoch_result = merge_dicts(
            dicts=val_step_results,
            default_func=lambda x: torch.stack(x).mean().item()
        )

        return validation_epoch_result

    def configure_optimizers(self):
        optimizer = self._get_optimizer()
        scheduler = self._get_lr_scheduler(optimizer=optimizer)

        if scheduler is None:
            return optimizer
        else:
            return [optimizer], [scheduler]

    def _step(self, model_inp: ModelInput) -> Tuple[torch.Tensor, Mapping]:
        output = self.forward(model_inp=model_inp)
        log = self._get_step_log(model_output=output)
        return output.loss, log

    @abc.abstractmethod
    def _get_optimizer(self):
        pass

    @abc.abstractmethod
    def _get_lr_scheduler(self, optimizer):
        pass

    @abc.abstractmethod
    def _get_step_log(self, model_output: ModelOutput) -> Dict:
        pass

    @abc.abstractmethod
    def get_description(self) -> Dict:
        pass
