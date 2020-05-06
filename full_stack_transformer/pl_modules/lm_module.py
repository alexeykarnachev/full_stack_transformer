import argparse
import json
from dataclasses import dataclass

import pytorch_lightning as pl
import torch
import transformers
from cached_property import cached_property
from pytorch_lightning.loggers.base import merge_dicts
from torch.utils.data.dataloader import DataLoader
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

import full_stack_transformer.tokenization
from full_stack_transformer.datasets.documents_dataset import load_from_dir
from full_stack_transformer.losses.unlikelihood_candidates_loss import \
    unlikelihood_candidates_loss
from full_stack_transformer.pl_modules.model_loading import \
    load_transformer_model_from_path
from full_stack_transformer.text_generator.text_generator import \
    TextGenerator, TextGeneratorParams
from full_stack_transformer.utilities.file_io import load_json, dump_json


class LMModule(pl.LightningModule):

    @cached_property
    def _generated_samples_file(self):
        return self.hparams.experiment_dir / 'generated.txt'

    @cached_property
    def _tokenizer_cls_name(self):
        description = load_json(self.hparams.dataset_dir / 'description.json')
        return description['tokenizer_cls_name']

    @cached_property
    def _text_generator_params_file(self):
        return self.hparams.experiment_dir / 'text_generator_params.json'

    @cached_property
    def _default_text_generator_params(self):
        params = TextGeneratorParams(
            seed_text=None,
            ignored_words=None,
            generation_max_len=64,
            temperature=0.7,
            top_k=50,
            top_p=1.0,
            repetition_penalty=5.0,
            num_return_sequences=16)
        return params

    @property
    def current_lr(self):
        return self.trainer.optimizers[0].param_groups[0]['lr']

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.hparams = hparams
        self.tokenizer = full_stack_transformer.tokenization.get_tokenizer(
            self._tokenizer_cls_name)

        self.model = load_transformer_model_from_path(
            model_path=str(hparams.model_path),
            vocab_size=self.tokenizer.get_vocab_size())

        self.hparams.transformer_config = self.model.config

        dump_json(
            obj=dict(self._default_text_generator_params),
            file_path=self._text_generator_params_file)

        self._dataset_dir = self.hparams.dataset_dir
        self._batch_size = self.hparams.batch_size
        self._learning_rate = self.hparams.learning_rate
        self._num_warmup_steps = self.hparams.num_warmup_steps
        self._num_cycles = self.hparams.num_cycles
        self._max_epochs = self.hparams.max_epochs
        self._unlikelihood_alpha = self.hparams.unlikelihood_alpha

    def prepare_data(self) -> None:
        for name in ['train', 'valid']:
            dataset = load_from_dir(self._dataset_dir / name)
            setattr(self, f'_{name}_dataset', dataset)

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader('train')

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader('valid')

    def _get_dataloader(self, name: str):
        dataset = getattr(self, f'_{name}_dataset')
        dataloader = dataset.get_dataloader(
            batch_size=self._batch_size,
            pad_val=self.tokenizer.get_pad_token_id())
        return dataloader

    def forward(self, documents_batch) -> 'ForwardResult':
        model_output = self.model(
            input_ids=documents_batch,
            labels=documents_batch)

        mle_loss = model_output[0]
        logits = model_output[1]

        ul_loss = unlikelihood_candidates_loss(
            logits=logits,
            target=documents_batch)

        loss = mle_loss + self._unlikelihood_alpha * ul_loss

        forward_res = ForwardResult(
            mle_loss=mle_loss,
            ul_loss=ul_loss,
            loss=loss)

        return forward_res

    def training_step(self, batch, batch_idx):
        forward_res = self.forward(batch)

        log = {
            'MLE-Loss/train': forward_res.mle_loss,
            'UL-Loss/train': forward_res.ul_loss,
            'Loss/train': forward_res.loss,
            'Learning-Rate': self.current_lr}

        return {'loss': forward_res.loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        forward_res = self.forward(batch)

        validation_step_result = {
            'log': {
                'MLE-Loss/valid': forward_res.mle_loss,
                'UL-Loss/valid': forward_res.ul_loss,
                'Loss/valid': forward_res.loss},
            'val_loss': forward_res.loss}

        return validation_step_result

    def validation_epoch_end(self, outputs):
        self._generate_and_log_text_samples()
        validation_epoch_result = merge_dicts(
            outputs, default_func=lambda x: torch.stack(x).mean())

        return validation_epoch_result

    def _generate_and_log_text_samples(self):
        generator = TextGenerator(
            model=self.model, tokenizer=self.tokenizer)

        params, error_msg = self._get_text_generator_params()
        text_samples = generator(params)
        self._write_generation_result(text_samples, params, error_msg)

    def _write_generation_result(self, text_samples, params, error_msg):
        dict_to_write = {
            'Global step': self.trainer.global_step,
            'Current epoch': self.trainer.current_epoch,
            'Generator params': dict(params),
            'Error message': error_msg or None,
            'Generated samples': text_samples}

        with open(self._generated_samples_file, 'a') as file:
            string_to_write = json.dumps(
                obj=dict_to_write, ensure_ascii=False, indent=4)
            string_to_write += '\n'
            file.write(string_to_write)

    def _get_text_generator_params(self):
        try:
            params = load_json(self._text_generator_params_file)
            params = TextGeneratorParams(**params)
            error_msg = ''
        except Exception as e:
            params = self._default_text_generator_params
            error_msg = str(e)

        return params, error_msg

    def configure_optimizers(self):
        optimizer = self._get_optimizer()
        scheduler = self._get_lr_scheduler(optimizer)

        return [optimizer], [scheduler]

    def _get_optimizer(self):
        parameters = self.model.parameters()
        optimizer = transformers.AdamW(
            params=parameters,
            lr=self._learning_rate)

        return optimizer

    def _get_lr_scheduler(self, optimizer):

        lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self._num_warmup_steps,
            num_training_steps=self._get_num_training_steps(),
            num_cycles=self._num_cycles)
        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': self.trainer.accumulate_grad_batches,
            'monitor': 'Loss/valid'}

        return scheduler

    def _get_num_training_steps(self):
        total_steps = len(self.train_dataloader()) * self._max_epochs
        training_steps = total_steps // self.trainer.accumulate_grad_batches
        return training_steps


@dataclass
class ForwardResult:
    mle_loss: torch.Tensor
    ul_loss: torch.Tensor
    loss: torch.Tensor
