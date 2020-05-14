import json
import pathlib
from collections import Mapping

from pytorch_lightning import Callback, Trainer

from full_stack_transformer.core.modelling.lightning import PLModule
from full_stack_transformer.core.text_input import TextInput
from full_stack_transformer.tasks.common.language_generator.generator import LanguageGeneratorParams, \
    LanguageGenerator
from full_stack_transformer.tasks.common.text_inputs.document import DocumentInput
from full_stack_transformer.utilities.experiment import Workspace


class LanguageGeneratorCallback(Callback):
    _OUTPUT_FILE_NAME = 'generated.txt'

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

    def __init__(
            self,
            experiment_workspace: Workspace,
            text_input: TextInput
    ):
        self._workspace = experiment_workspace
        self._text_input = text_input

    def on_validation_end(
            self,
            trainer: Trainer,
            pl_module: PLModule
    ):
        params = self._default_params
        tokenizer = pl_module.tokenizer
        model = pl_module.model

        generator = LanguageGenerator(
            model=model,
            eos_token_id=tokenizer.eos_token_id
        )

        inp_encoding = pl_module.tokenizer.encode_for_inference(
            text_input=self._text_input,
        )[0]

        encodings = generator(encoding=inp_encoding, params=params)

        text_samples = [tokenizer.decode_encoding(e) for e in encodings]

        result = {
            'Global step': trainer.global_step,
            'Current epoch': trainer.current_epoch,
            'Generator params': params.__dict__,
            'Generated samples': text_samples
        }

        self._dump_result(result=result)

    def _dump_result(self, result: Mapping):
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
            text_input=text_input
        )
