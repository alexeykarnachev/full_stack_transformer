from typing import Dict

import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW

from full_stack_transformer.core.model_input import ModelInput
from full_stack_transformer.core.model_output import ModelOutput
from full_stack_transformer.core.modelling.lightning import PLModule
from full_stack_transformer.core.modelling.model import Model


class _FakeModel(Model):
    def __init__(self):
        super().__init__()
        self._loss = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self._logits = torch.tensor(0)

    def forward(self, inp: ModelInput) -> ModelOutput:
        return ModelOutput(loss=self._loss, logits=self._logits)

    def infer(self, inp: ModelInput) -> ModelOutput:
        return ModelOutput(loss=None, logits=self._logits)


class _FakePLModule(PLModule):
    def __init__(self, model: Model):
        super().__init__(model)
        self.step_log = None

    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.rand(64, 10)), batch_size=16)

    def val_dataloader(self):
        return DataLoader(TensorDataset(torch.rand(64, 10)), batch_size=16)

    def _get_optimizer(self):
        return AdamW(self.model.parameters(), lr=100)

    def _get_lr_scheduler(self, optimizer):
        return None

    def _get_step_log(self, model_output: ModelOutput) -> Dict:
        self.step_log = {'loss': model_output.loss}
        return self.step_log

    def get_description(self) -> Dict:
        return dict()


def test_pl_module():
    model = _FakeModel()
    module = _FakePLModule(model=model)
    trainer = Trainer(
        max_epochs=3,
        num_sanity_val_steps=0,
        logger=False,
        checkpoint_callback=False
    )
    trainer.fit(module)

    assert module.step_log['loss'] < -100
