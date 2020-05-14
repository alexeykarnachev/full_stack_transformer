from typing import List

from pytorch_lightning import Callback

from full_stack_transformer.core.task_runner import TaskRunner
from full_stack_transformer.tasks.common.callbacks.language_generator import \
    LanguageGeneratorDocumentCallback
from full_stack_transformer.tasks.document_decoder.lightning import DocumentDecPLModule


class DocumentDecTaskRunner(TaskRunner):
    def __init__(self):
        super().__init__(pl_module_cls=DocumentDecPLModule)

    def _get_trainer_callbacks(self) -> List[Callback]:
        generator = LanguageGeneratorDocumentCallback(
            experiment_workspace=self.workspace
        )
        return [generator]


if __name__ == '__main__':
    runner = DocumentDecTaskRunner()
    runner.run()
