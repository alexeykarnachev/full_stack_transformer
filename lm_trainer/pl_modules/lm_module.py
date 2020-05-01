import argparse
import json

import pytorch_lightning as pl
import torch
import transformers
from cached_property import cached_property
from torch.utils.data.dataloader import DataLoader
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

import lm_trainer.tokenization
from lm_trainer.datasets.documents_dataset import load_from_dir
from lm_trainer.pl_modules.model_loading import load_transformer_model_from_path
from lm_trainer.text_generator.text_generator import TextGenerator, TextGeneratorParams
from lm_trainer.utilities.file_io import load_json, dump_json


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

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.hparams = hparams
        self.tokenizer = lm_trainer.tokenization.get_tokenizer(
            self._tokenizer_cls_name)

        self.model = load_transformer_model_from_path(
            model_path=str(hparams.model_path),
            vocab_size=self.tokenizer.get_vocab_size())

        self.hparams.transformer_config = self.model.config

        dump_json(
            obj=dict(self._default_text_generator_params),
            file_path=self._text_generator_params_file)

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
            pad_val=self.tokenizer.get_pad_token_id())
        return dataloader

    def forward(self, documents_batch):
        output = self.model(documents_batch, labels=documents_batch)
        return output

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)[0]
        lr = self.trainer.optimizers[0].param_groups[0]['lr']

        log = {'Loss/train': loss, 'Learning-Rate': lr}

        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        validation_step_result = {'val_loss': output[0]}
        return validation_step_result

    def validation_epoch_end(self, outputs):
        outputs_processor = ValidationEpochResultsProcessor(
            validation_outputs=outputs)

        if self.hparams.log_text_samples:
            self._generate_and_log_text_samples()

        loss = outputs_processor.get_validation_loss()
        logs = {'Loss/valid': loss}

        return {'val_loss': loss, 'log': logs}

    def _generate_and_log_text_samples(self):
        generator = TextGenerator(
            model=self.model,
            eos_token_id=self.tokenizer.get_eos_token_id(),
            tokenizer=self.tokenizer)

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


class ValidationEpochResultsProcessor:
    def __init__(self, validation_outputs, ):
        self._outputs = validation_outputs

    def get_validation_loss(self):
        losses = [out['val_loss'] for out in self._outputs]
        loss = torch.stack(losses).mean()

        return loss
