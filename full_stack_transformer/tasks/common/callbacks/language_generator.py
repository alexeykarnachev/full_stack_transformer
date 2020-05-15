import json
import pathlib
from collections import Mapping
from typing import Tuple, List, Optional

from pytorch_lightning import Callback, Trainer

from full_stack_transformer.core.modelling.lightning import PLModule
from full_stack_transformer.core.text_input import TextInput
from full_stack_transformer.tasks.common.language_generator.generator import (
    LanguageGeneratorParams,
    LanguageGenerator
)
from full_stack_transformer.tasks.common.text_inputs.dialog import DialogInput
from full_stack_transformer.tasks.common.text_inputs.document import DocumentInput
from full_stack_transformer.utilities.experiment import Workspace


class LanguageGeneratorCallback(Callback):
    _OUTPUT_FILE_NAME = 'generated.txt'
    _CONFIG_FILE_NAME = 'generation_config.json'

    @property
    def _default_params(self):
        return LanguageGeneratorParams(
            max_number_of_generated_tokens=128,
            num_return_sequences=8,
            repetition_penalty=1.0,
            temperature=0.7,
            top_k=0,
            top_p=1.0
        )

    @property
    def _out_file(self) -> pathlib.Path:
        return self._workspace.experiment_dir / self._OUTPUT_FILE_NAME

    @property
    def _cfg_file(self) -> pathlib.Path:
        return self._workspace.experiment_dir / self._CONFIG_FILE_NAME

    def __init__(
            self,
            experiment_workspace: Workspace,
            default_text_input: TextInput
    ):
        self._workspace = experiment_workspace
        self._dflt_text_input = default_text_input

        self._save_config()

    def _save_config(self):
        generator_params = self._default_params.__dict__
        text_input_params = self._dflt_text_input.__dict__

        cfg = {
            'generator_params': generator_params,
            'text_input_params': text_input_params
        }
        cfg = [cfg]

        with self._cfg_file.open('w') as file:
            json.dump(cfg, file, ensure_ascii=False, indent=2)

    def _load_params_and_inputs(
            self
    ) -> Tuple[List[Tuple[LanguageGeneratorParams, TextInput]], Optional[str]]:
        try:
            output = []
            with self._cfg_file.open() as file:
                cfg = json.load(file)

            for c in cfg:
                params = LanguageGeneratorParams(**c['generator_params'])
                inp = self._dflt_text_input.__class__(**c['text_input_params'])
                output.append((params, inp))

            err = None
        except Exception as e:
            output = [(self._default_params, self._dflt_text_input)]
            err = str(e)

        return output, err

    def on_validation_end(
            self,
            trainer: Trainer,
            pl_module: PLModule
    ):
        params_and_inputs, err = self._load_params_and_inputs()
        results = []
        for params, inp in params_and_inputs:
            tokenizer = pl_module.tokenizer
            model = pl_module.model

            generator = LanguageGenerator(
                model=model,
                eos_token_id=tokenizer.eos_token_id
            )

            inp_encoding = pl_module.tokenizer.encode_for_inference(inp)
            encodings = generator(encoding=inp_encoding, params=params)
            text_samples = [tokenizer.decode_encoding(e) for e in encodings]

            result = {
                'Global step': trainer.global_step,
                'Current epoch': trainer.current_epoch,
                'Error message': err,
                'Generator params': params.__dict__,
                'Generator input': inp.__dict__,
                'Generated samples': text_samples,
            }
            results.append(result)

        self._dump_result(result=results)

    def _dump_result(self, result: List):
        with self._out_file.open('a') as file:
            out_str = json.dumps(obj=result, ensure_ascii=False, indent=4)
            out_str += '\n'
            file.write(out_str)


class LanguageGeneratorDocumentCallback(LanguageGeneratorCallback):
    def __init__(
            self,
            experiment_workspace: Workspace
    ):
        text_input = DocumentInput(body='', meta=None)
        super().__init__(
            experiment_workspace=experiment_workspace,
            default_text_input=text_input
        )


class LanguageGeneratorDialogCallback(LanguageGeneratorCallback):
    def __init__(
            self,
            experiment_workspace: Workspace
    ):
        text_input = DialogInput(
            utterances=[''],
            persona=None,
            persona_idx=None,
            tags=None
        )
        super().__init__(
            experiment_workspace=experiment_workspace,
            default_text_input=text_input
        )
