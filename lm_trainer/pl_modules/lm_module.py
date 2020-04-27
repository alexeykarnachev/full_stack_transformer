import argparse

import pytorch_lightning as pl
import torch
import transformers
from torch.utils.data.dataloader import DataLoader
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

import lm_trainer.tokenizers
from lm_trainer.datasets.documents_dataset import load_from_dir


class LMModule(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()

        self._model = transformers.GPT2LMHeadModel.from_pretrained(
            hparams.model_path
        )
        self._tokenizer = lm_trainer.tokenizers.get_tokenizer(
            hparams.tolenizer_cls_name
        )

    def prepare_data(self) -> None:
        for name in ['train', 'valid']:
            dataset = load_from_dir(self.hparams.dataset_dir / name)
            setattr(self, f'_{name}_dataset', dataset)

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader('train')

    def valid_dataloader(self) -> DataLoader:
        return self._get_dataloader('valid')

    def _get_dataloader(self, name: str):
        dataset = getattr(self, f'_{name}_dataset')
        dataloader = dataset.get_dataloader(
            batch_size=self.hparams.batch_size,
            pad_val=self.hparams.pad_val)
        return dataloader

    def forward(self, documents_batch):
        loss = self._model(documents_batch)[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']

        log = {'Loss/train': loss, 'Learning-Rate': lr}

        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        logs = {'Loss/valid': loss}

        return {'val_loss': loss, 'log': logs}

    def configure_optimizers(self):
        optimizer = self._get_optimizer()
        scheduler = self._get_lr_scheduler(optimizer)

        return [optimizer], [scheduler]

    def _get_optimizer(self):
        parameters = self._model.parameters()
        optimizer = transformers.AdamW(
            params=parameters,
            lr=self.hparams.learning_rate)

        return optimizer

    def _get_lr_scheduler(self, optimizer):
        lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self._get_num_training_steps(),
            num_cycles=self.hparams.num_cycles
        )
        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': self.trainer.accumulate_grad_batches,
            'monitor': 'Loss/valid'
        }

        return scheduler

    def _get_num_training_steps(self):
        total_steps = len(self.train_dataloader()) * self.hparams.n_epochs
        training_steps = total_steps // self.trainer.accumulate_grad_batches
        return training_steps
