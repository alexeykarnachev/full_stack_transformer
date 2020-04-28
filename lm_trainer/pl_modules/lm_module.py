import argparse

import pytorch_lightning as pl
import torch
import transformers
from torch.utils.data.dataloader import DataLoader
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

import lm_trainer.tokenizers
from lm_trainer.datasets.documents_dataset import load_from_dir
from lm_trainer.utilities.file_io import load_json


class LMModule(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.hparams = hparams
        self._tokenizer = lm_trainer.tokenizers.get_tokenizer(
            self._get_tokenizer_cls_name())

        self._model = get_gpt_model(
            model_path=hparams.model_path,
            vocab_size=self._tokenizer.get_vocab_size())

    def _get_tokenizer_cls_name(self):
        description = load_json(self.hparams.dataset_dir / 'description.json')
        return description['tokenizer_cls_name']

    def prepare_data(self) -> None:
        for name in ['train', 'valid']:
            dataset = load_from_dir(self.hparams.dataset_dir / name)
            setattr(self, f'_{name}_dataset', dataset)

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader('train')

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader('valid')

    def _get_dataloader(self, name: str):
        dataset = getattr(self, f'_{name}_dataset')
        dataloader = dataset.get_dataloader(
            batch_size=self.hparams.batch_size,
            pad_val=self._tokenizer.get_pad_val())
        return dataloader

    def forward(self, documents_batch):
        loss = self._model(documents_batch, labels=documents_batch)[0]
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
        total_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        training_steps = total_steps // self.trainer.accumulate_grad_batches
        return training_steps


def get_gpt_model(model_path, vocab_size: int):
    model = transformers.GPT2LMHeadModel.from_pretrained(model_path)
    model.resize_token_embeddings(vocab_size)
    return model
