import abc
import argparse
import copy
from typing import Dict, Type, List

from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from full_stack_transformer.core.modelling.lightning import PLModule
from full_stack_transformer.utilities.experiment import Workspace


class TaskRunner(abc.ABC):
    def __init__(self, pl_module_cls: Type[PLModule]):
        self.pl_module_cls = pl_module_cls
        args = self._parse_args()

        self.workspace = Workspace(
            experiments_root=args['experiments_root'],
            experiment_name=args['experiment_name']
        )

    def _parse_args(self) -> Dict:
        parser = argparse.ArgumentParser()
        parser = Trainer.add_argparse_args(parser)
        parser = Workspace.add_argparse_args(parser)
        parser = self.pl_module_cls.add_argparse_args(parser)

        args = parser.parse_args()

        return args.__dict__

    def run(self):
        args = self._parse_args()

        module = self.pl_module_cls(**args)

        description = {
            'Arguments': args,
            'Module': module.get_description()
        }
        self.workspace.save_description(description)

        trainer = self._prepare_trainer(args=args)

        trainer.fit(model=module)

    def _prepare_trainer(self, args):
        trainer_args = copy.deepcopy(args)

        _fix_trainer_args(args=trainer_args)

        trainer_args.update(
            {
                'logger': TensorBoardLogger(
                    save_dir=self.workspace.logs_dir,
                    name=self.workspace.name,
                    version=self.workspace.version
                ),
                'checkpoint_callback': ModelCheckpoint(
                    filepath=self.workspace.models_dir,
                    verbose=True,
                    save_top_k=2,
                    period=0
                )
            }
        )

        trainer_args['callbacks'] = self._get_trainer_callbacks()

        trainer = Trainer(**trainer_args)

        return trainer

    @abc.abstractmethod
    def _get_trainer_callbacks(self) -> List[Callback]:
        pass


def _fix_trainer_args(args: Dict) -> None:
    val_check_interval = args['val_check_interval']
    if val_check_interval <= 1:
        val_check_interval = float(val_check_interval)
    else:
        val_check_interval = int(val_check_interval)

    args['val_check_interval'] = val_check_interval
