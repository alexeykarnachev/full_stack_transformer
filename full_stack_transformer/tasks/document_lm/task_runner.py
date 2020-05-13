from typing import List

from pytorch_lightning import Callback

from full_stack_transformer.core.task_runner import TaskRunner
from full_stack_transformer.tasks.document_lm.language_generator.callback import LanguageGeneratorCallback
from full_stack_transformer.tasks.document_lm.modelling.lightning import DocumentPLModule


class DocumentLMTaskRunner(TaskRunner):
    def __init__(self, experiments_root: str, experiment_name: str):
        super().__init__(
            experiments_root=experiments_root,
            experiment_name=experiment_name,
            pl_module_cls=DocumentPLModule
        )

    def _get_trainer_callbacks(self) -> List[Callback]:
        return [LanguageGeneratorCallback(experiment_workspace=self.workspace)]