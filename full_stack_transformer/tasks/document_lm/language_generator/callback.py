import json
import pathlib
from collections import Mapping

from pytorch_lightning import Callback, Trainer

from full_stack_transformer.tasks.document_lm.language_generator.generator import LanguageGeneratorParams, \
    LanguageGenerator
from full_stack_transformer.tasks.document_lm.modelling.lightning import DocumentPLModule
from full_stack_transformer.tasks.document_lm.text_input import DocumentInput
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
    def _default_document(self):
        return DocumentInput(body='', meta=None)

    @property
    def _out_file(self) -> pathlib.Path:
        return self._workspace.experiment_dir / self._OUTPUT_FILE_NAME

    def __init__(self, experiment_workspace: Workspace):
        self._workspace = experiment_workspace

    def on_validation_end(
            self,
            trainer: Trainer,
            pl_module: DocumentPLModule
    ):
        params = self._default_params
        tokenizer = pl_module.tokenizer
        generator = LanguageGenerator(
            model=pl_module.model,
            eos_token_id=tokenizer.eos_token_id
        )

        inp_encoding = tokenizer.encode_for_inference(
            text_input=self._default_document,
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
