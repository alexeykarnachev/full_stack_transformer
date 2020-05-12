import pathlib
from typing import Dict

import transformers
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

from full_stack_transformer.core.model_output import ModelOutput
from full_stack_transformer.core.modelling.lightning import PLModule
from full_stack_transformer.tasks.document_lm.modelling.model import \
    DocumentModel


class DocumentPLModule(PLModule):
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
        self._max_meta_len = max_meta_len
        self._max_body_len = max_body_len
        self._ignore_meta_prob = ignore_meta_prob
        self._model_path = model_path
        self.train_file = pathlib.Path(train_file)
        self.valid_file = pathlib.Path(valid_file)
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
        self._model = DocumentModel(
            lm_head_model=self._gpt,
            unlikelihood_alpha=self._unlikelihood_alpha
        )

        self._train_dataset = None
        self._valid_dataset = None

        locals_ = locals()

        gpt_config = json.dumps(self.transformer_config, ensure_ascii=False)
        self.hparams = _get_hparams(
            locals_=locals_,
            transformer_config=gpt_config
        )

        super().__init__(model=self.model)

    def _get_optimizer(self):
        parameters = self._model.parameters()
        optimizer = transformers.AdamW(
            params=parameters,
            lr=self.learning_rate
        )

        return optimizer

    def _get_lr_scheduler(self, optimizer):
        lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
            num_cycles=self.num_cycles
        )
        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': self.trainer.accumulate_grad_batches,
            'monitor': 'Loss/valid'
        }

        return scheduler

    def _get_step_log(self, model_output: ModelOutput) -> Dict:
        pass