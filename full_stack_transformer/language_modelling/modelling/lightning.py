import inspect
import pathlib
from argparse import Namespace
from typing import Mapping, Tuple, Dict, Sequence

import torch
import transformers
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.base import merge_dicts
from torch.utils.data.dataloader import DataLoader
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

from full_stack_transformer.language_modelling.data_structures import (
    LanguageModelInput,
    LanguageModelOutput
)
from full_stack_transformer.language_modelling.documents_dataset import \
    DocumentDataset
from full_stack_transformer.language_modelling.modelling.loading import \
    load_transformer_model_from_path
from full_stack_transformer.language_modelling.modelling.model import \
    LanguageModel
from full_stack_transformer.language_modelling.tokenization.tokenizer import (
    get_tokenizer,
    DocumentTokenizer
)
from full_stack_transformer.utilities.cli import ArgparserExtender


class LanguagePLModule(LightningModule, ArgparserExtender):
    @property
    def _current_lr(self):
        return self.trainer.optimizers[0].param_groups[0]['lr']

    @property
    def transformer_config(self):
        return self._gpt.config.__dict__

    @property
    def model(self) -> LanguageModel:
        return self._model

    @property
    def tokenizer(self) -> DocumentTokenizer:
        return self._tokenizer

    def __init__(
            self,
            model_path: str,
            tokenizer_class_name: str,
            batch_size: int,
            max_meta_len: int,
            max_body_len: int,
            ignore_meta_prob: float,
            train_file: str,
            valid_file: str,
            learning_rate: float,
            num_warmup_steps: int,
            num_cycles: int,
            unlikelihood_alpha: float,
            **kwargs
    ):
        """
        Args:
            model_path (str):
                Path to the pre-trained transformer model.

            tokenizer_class_name (str):
                Name of the tokenizer class which is importable from
                `dialog_models.dialog_data`.

            batch_size (int):
                Batch size (the same for training and validation).

            max_meta_len (int):
                Max number of tokens for `meta` field encoding. Longer meta
                field will be cut from the left side (right side will be
                remained).

            max_body_len (int):
                Max number of tokens for `body` field encoding. Longer bodies
                will be chunked. Encoding of the `meta` will be appended to
                each part of the encoded body.

            ignore_meta_prob (float):
                The probability to ignore `meta` field in the document and don't
                add it to the final encoding.

            train_file (str):
                Path to the training raw dialog samples file.

            valid_file (str):
                Path to the validation raw dialog samples file.

            learning_rate (float):
                Base learning rate for AdamW optimizer.

            num_warmup_steps (int):
                Number of warmup steps for the cosine with hard restarts
                scheduler.

            num_cycles (int):
                Number of cycles for the cosine with hard restarts scheduler.
                If 0, the scheduler will perform as a constant scheduler with
                warmup.

            unlikelihood_alpha (float):
                Unlikelihood loss multiplier. If None, no unlikelihood loss will
                be used.
        """
        super().__init__()

        self._tokenizer_class_name = tokenizer_class_name
        self._batch_size = batch_size
        self._max_meta_len = max_meta_len
        self._max_body_len = max_body_len
        self._ignore_meta_prob = ignore_meta_prob
        self._model_path = model_path
        self._train_file = pathlib.Path(train_file)
        self._valid_file = pathlib.Path(valid_file)
        self._unlikelihood_alpha = unlikelihood_alpha

        self._learning_rate = learning_rate
        self._num_warmup_steps = num_warmup_steps
        self._num_cycles = num_cycles

        self._tokenizer = get_tokenizer(
            name=self._tokenizer_class_name,
            max_meta_len=self._max_meta_len,
            max_body_len=self._max_body_len,
            ignore_meta_prob=self._ignore_meta_prob
        )
        self._gpt = load_transformer_model_from_path(
            model_path=self._model_path,
            vocab_size=self._tokenizer.vocab_size
        )
        self._model = LanguageModel(
            lm_head_model=self._gpt,
            unlikelihood_alpha=self._unlikelihood_alpha
        )

        self._train_dataset = None
        self._valid_dataset = None

        locals_ = locals()
        self.hparams = _get_hparams(locals_=locals_)

    def prepare_data(self) -> None:
        self._train_dataset = DocumentDataset(
            file_path=self._train_file,
            tokenizer=self._tokenizer,
            n_producer_workers=4,
            chunk_size=2000
        )
        self._valid_dataset = DocumentDataset(
            file_path=self._valid_file,
            tokenizer=self._tokenizer,
            n_producer_workers=1,
            chunk_size=2000
        )

    def train_dataloader(self) -> DataLoader:
        return self._train_dataset.get_data_loader(
            batch_size=self._batch_size,
            pad_value=self._tokenizer.pad_token_id,
            num_workers=4
        )

    def val_dataloader(self) -> DataLoader:
        return self._valid_dataset.get_data_loader(
            batch_size=self._batch_size,
            pad_value=self._tokenizer.pad_token_id
        )

    def forward(self, model_inp: LanguageModelInput) -> LanguageModelOutput:
        output = self._model(model_inp)
        return output

    def training_step(
            self,
            model_inp: LanguageModelInput,
            batch_idx: int
    ) -> Dict:
        loss, log = self._step(model_inp=model_inp)
        return {'loss': loss, 'log': log}

    def validation_step(
            self,
            model_inp: LanguageModelInput,
            batch_idx: int
    ) -> Dict:
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

        return [optimizer], [scheduler]

    def _get_optimizer(self):
        parameters = self._model.parameters()
        optimizer = transformers.AdamW(
            params=parameters,
            lr=self._learning_rate
        )

        return optimizer

    def _get_lr_scheduler(self, optimizer):
        lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self._num_warmup_steps,
            num_training_steps=self._get_num_training_steps(),
            num_cycles=self._num_cycles
        )
        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': self.trainer.accumulate_grad_batches,
            'monitor': 'Loss/valid'
        }

        return scheduler

    def _get_num_training_steps(self):
        total_steps = len(self.train_dataloader()) * self.trainer.max_epochs
        training_steps = total_steps // self.trainer.accumulate_grad_batches
        return training_steps

    def _step(
            self,
            model_inp: LanguageModelInput
    ) -> Tuple[torch.Tensor, Mapping]:
        output = self.forward(model_inp=model_inp)
        log = self._get_step_log(model_output=output)
        return output.loss, log

    def _get_step_log(self, model_output: LanguageModelOutput):
        postfix = 'train' if self.training else 'valid'
        log = {
            f'LM-Loss/{postfix}': model_output.lm_loss,
            f'Loss/{postfix}': model_output.loss
        }

        if self.training:
            log['Learning-Rate'] = self._current_lr

        return log


def _get_hparams(locals_):
    signature = inspect.signature(LanguagePLModule.__init__)

    params = {}

    for name, param in signature.parameters.items():
        value = locals_[name]
        if name == 'self':
            continue
        elif name == 'kwargs':
            params.update(value)
        else:
            params[name] = value

    return Namespace(**params)
