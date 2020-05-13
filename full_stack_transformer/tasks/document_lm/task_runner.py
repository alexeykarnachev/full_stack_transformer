from typing import List

from pytorch_lightning import Callback

from full_stack_transformer.core.task_runner import TaskRunner
from full_stack_transformer.tasks.document_lm.language_generator.callback import LanguageGeneratorCallback
from full_stack_transformer.tasks.document_lm.modelling.lightning import DocumentPLModule


class DocumentLMTaskRunner(TaskRunner):
    def __init__(self):
        super().__init__(pl_module_cls=DocumentPLModule)

    def _get_trainer_callbacks(self) -> List[Callback]:
        return [LanguageGeneratorCallback(experiment_workspace=self.workspace)]


if __name__ == '__main__':
    runner = DocumentLMTaskRunner()
    runner.run()