from typing import List

from pytorch_lightning import Callback

from full_stack_transformer.core.task_runner import TaskRunner
from full_stack_transformer.tasks.dialog_decoder.lightning import DialogDecPLModule


class DialogDecTaskRunner(TaskRunner):
    def __init__(self):
        super().__init__(pl_module_cls=DialogDecPLModule)

    def _get_trainer_callbacks(self) -> List[Callback]:
        return list()


if __name__ == '__main__':
    runner = DialogDecTaskRunner()
    runner.run()
