import argparse
import pathlib
import re
from typing import Sequence

import more_itertools
import pytorch_lightning as pl
import torch
import transformers
from torch.utils.data.dataloader import DataLoader
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

import lm_trainer.tokenizers
from lm_trainer.datasets.documents_dataset import load_from_dir
from lm_trainer.text_generator.text_generator import TextGenerator, TextGeneratorParams
from lm_trainer.utilities.file_io import load_json


class LMModule(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.hparams = hparams
        self._tokenizer = lm_trainer.tokenizers.get_tokenizer(
            self._get_tokenizer_cls_name())

        self._model = get_gpt_model_from_model_path(
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
            pad_val=self._tokenizer.get_pad_token_id())
        return dataloader

    def forward(self, documents_batch):
        output = self._model(documents_batch, labels=documents_batch)
        return output

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)[0]
        lr = self.trainer.optimizers[0].param_groups[0]['lr']

        log = {'Loss/train': loss, 'Learning-Rate': lr}

        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        last_hidden_states = output[-1][-1]

        validation_step_result = {
            'val_loss': output[0],
            'last_hidden_states': last_hidden_states,
            'token_ids': batch
        }
        return validation_step_result

    def validation_epoch_end(self, outputs):
        outputs_processor = ValidationEpochResultsProcessor(
            validation_outputs=outputs,
            max_n_samples_to_embed=1000,
            max_n_tokens_to_embed=24,
            tokenizer=self._tokenizer)

        self._log_extra_info(outputs_processor)
        loss = outputs_processor.get_validation_loss()
        logs = {'Loss/valid': loss}

        return {'val_loss': loss, 'log': logs}

    def _log_extra_info(self, outputs_processor):
        if self.hparams.log_embeddings:
            self._log_embeddings(outputs_processor)
        if self.hparams.log_text_samples:
            self._log_text_samples()

    def _log_embeddings(self, outputs_processor):
        embeddings = outputs_processor.get_validation_embeddings()
        texts = outputs_processor.get_validation_texts()

        self.trainer.logger.experiment.add_embedding(
            mat=embeddings,
            metadata=texts,
            global_step=self.trainer.global_step)

    def _generate_text_samples(self):
        generator = TextGenerator(
            model=self._model,
            eos_token_id=self._tokenizer.get_eos_token_id(),
            tokenizer=self._tokenizer)

        text_generator_params = TextGeneratorParams(
            seed_token_ids=None,
            ignored_token_ids=None,
            generation_max_len=36,
            temperature=0.7,
            top_k=50,
            top_p=1.0,
            repetition_penalty=5.0,
            num_return_sequences=10)

        generated_text_samples = generator(text_generator_params)

        return generated_text_samples

    def _log_text_samples(self):
        text_samples = self._generate_text_samples()
        file_path: pathlib.Path = self.hparams.experiment_dir / 'generated.txt'

        texts = self._prepare_texts_for_logging(text_samples)

        with file_path.open('a') as file:
            file.write(texts)

    def _prepare_texts_for_logging(self, text_samples: Sequence[str]) -> str:
        step = self.trainer.global_step
        epoch = self.trainer.current_epoch
        header = f"Global step: {step}, Current epoch: {epoch}"
        texts_sep = '\n' + '-' * 80 + '\n'
        texts = texts_sep.join([header] + list(text_samples))
        texts += '\n\n\n' + '=' * 80 + '\n\n\n'
        return texts

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


class ValidationEpochResultsProcessor:
    def __init__(
            self,
            validation_outputs,
            max_n_tokens_to_embed,
            max_n_samples_to_embed,
            tokenizer
    ):
        self._outputs = validation_outputs
        self._max_n_tokens_to_embed = max_n_tokens_to_embed
        self._max_n_samples_to_embed = max_n_samples_to_embed
        self._tokenizer = tokenizer

    def get_validation_loss(self):
        losses = [out['val_loss'] for out in self._outputs]
        loss = torch.stack(losses).mean()

        return loss

    def get_validation_embeddings(self):
        hidden_states = [out['last_hidden_states'] for out in self._outputs]
        hidden_states = list(more_itertools.flatten(hidden_states))
        hidden_states = hidden_states[:self._max_n_samples_to_embed]
        hidden_states = [x[:self._max_n_tokens_to_embed] for x in hidden_states]
        embeddings = [x.mean(0) for x in hidden_states]
        embeddings = torch.stack(embeddings, 0)
        return embeddings

    def get_validation_texts(self):
        token_ids = []
        for out in self._outputs:
            ids = out['token_ids'].detach().cpu().numpy().tolist()
            token_ids.append(ids)

        token_ids = list(more_itertools.flatten(token_ids))
        token_ids = token_ids[:self._max_n_samples_to_embed]
        token_ids = [ids[:self._max_n_tokens_to_embed] for ids in token_ids]

        texts = self._tokenizer.decode_batch(token_ids)
        texts = [self._tokenizer.postprocess(text) for text in texts]
        texts = [re.sub(r'\s+', ' ', text) for text in texts]
        return texts


def get_gpt_model_from_model_path(model_path, vocab_size: int):
    model = transformers.GPT2LMHeadModel.from_pretrained(
        pretrained_model_name_or_path=model_path,
        output_past=True,
        output_hidden_states=True
    )
    model.resize_token_embeddings(vocab_size)

    return model
