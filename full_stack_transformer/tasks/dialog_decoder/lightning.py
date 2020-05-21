import json
import pathlib
from typing import Dict, Optional

import torch
import transformers
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

from full_stack_transformer.core.data.dataset import DataLoader
from full_stack_transformer.core.model_output import LanguageModelOutput
from full_stack_transformer.core.modelling.lightning import PLModule
from full_stack_transformer.core.modelling.loading import load_transformer_model_from_path
from full_stack_transformer.core.tokenizer import get_tokenizer
from full_stack_transformer.tasks.dialog_decoder.data.dataset import DialogDataset
from full_stack_transformer.tasks.common.models.hf_gpt2 import HFGPT2Model
from full_stack_transformer.tasks.document_decoder.data.dataset import DocumentDataset
from full_stack_transformer.utilities.arguments import get_func_arg_values_as_namespace


class DialogDecPLModule(PLModule):
    def __init__(
            self,
            model_path: str,
            tokenizer_class_name: str,
            batch_size: int,
            max_tags_len: int,
            max_pers_len: int,
            max_dialog_len: int,
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
            max_tags_len (int):
                Max number of tokens for `tags` field encoding. Longer tags
                field will be cut from the left side (right side will be
                remained).
            max_pers_len (int):
                Max number of tokens for `persona` field encoding. Longer
                persona field will be cut from the left side (right side will be
                remained).
            max_dialog_len (int):
                Max number of tokens for concatenated utterances (`utterances`
                field). Longer utterances will be cut from the left side (right
                side will be remained).
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
        self.train_file = pathlib.Path(train_file)
        self.valid_file = pathlib.Path(valid_file)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_warmup_steps = num_warmup_steps
        self.num_cycles = num_cycles

        tokenizer_config = dict(
            name=tokenizer_class_name,
            max_tags_len=max_tags_len,
            max_pers_len=max_pers_len,
            max_dialog_len=max_dialog_len
        )

        self.tokenizer = get_tokenizer(**tokenizer_config)

        lm_head_model = load_transformer_model_from_path(
            model_path=model_path,
            vocab_size=self.tokenizer.vocab_size
        )
        model = HFGPT2Model(
            lm_head_model=lm_head_model,
            unlikelihood_alpha=unlikelihood_alpha
        )

        self.train_dataset: Optional[DocumentDataset] = None
        self.valid_dataset: Optional[DocumentDataset] = None

        locals_ = locals()
        self.transformer_config = json.dumps(
            lm_head_model.config.__dict__,
            ensure_ascii=False
        )

        super().__init__(model=model)

        self.hparams = get_func_arg_values_as_namespace(
            locals_=locals_,
            func=self.__init__,
            transformer_config=self.transformer_config,
            tokenizer_config=tokenizer_config
        )

    def prepare_data(self) -> None:
        self.train_dataset = DialogDataset(
            file_path=self.train_file,
            tokenizer=self.tokenizer
        )

        self.valid_dataset = DialogDataset(
            file_path=self.valid_file,
            tokenizer=self.tokenizer
        )

    def train_dataloader(self) -> DataLoader:
        return self.train_dataset.get_data_loader(
            batch_size=self.batch_size,
            num_workers=4
        )

    def val_dataloader(self) -> DataLoader:
        return self.valid_dataset.get_data_loader(
            batch_size=self.batch_size,
            num_workers=1
        )

    def _get_optimizer(self):
        parameters = self.model.parameters()
        optimizer = transformers.AdamW(
            params=parameters,
            lr=self.learning_rate
        )

        return optimizer

    def _get_lr_scheduler(self, optimizer):
        total_steps = len(self.train_dataloader()) * self.trainer.max_epochs
        training_steps = total_steps // self.trainer.accumulate_grad_batches

        lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=training_steps,
            num_cycles=self.num_cycles
        )
        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': self.trainer.accumulate_grad_batches,
            'monitor': 'Loss/valid'
        }

        return scheduler

    def _get_step_log(self, model_output: LanguageModelOutput) -> Dict:
        postfix = 'train' if self.training else 'valid'
        log = {
            f'LM-Loss/{postfix}': model_output.lm_loss,
            f'UL-Loss/{postfix}': model_output.ul_loss,
            f'Loss/{postfix}': model_output.loss
        }

        if self.training:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            log['Learning-Rate'] = torch.tensor(
                current_lr, device=model_output.loss.device)

        return log

    def get_description(self) -> Dict:
        return {'Transformer': self.transformer_config}
