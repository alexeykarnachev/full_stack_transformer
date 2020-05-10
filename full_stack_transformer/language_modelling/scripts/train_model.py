import argparse
import copy
from typing import Dict

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from full_stack_transformer.language_generation.callback import LanguageGeneratorCallback
from full_stack_transformer.utilities.experiment import Workspace
from full_stack_transformer.language_modelling.modelling.lightning import LanguagePLModule


def main():
    args = _parse_args().__dict__

    workspace = Workspace(
        experiments_root=args['experiments_root'],
        experiment_name=args['experiment_name']
    )

    pl_module = LanguagePLModule(**args)

    workspace.save_description(
        content={
            'Experiment': args,
            'Transformer': pl_module.transformer_config
        }
    )

    trainer = _prepare_trainer(args=args, workspace=workspace)

    trainer.fit(model=pl_module)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = Workspace.add_argparse_args(parser)
    parser = LanguagePLModule.add_argparse_args(parser)

    args = parser.parse_args()
    return args


def _prepare_trainer(args: Dict, workspace: Workspace) -> Trainer:
    trainer_args = copy.deepcopy(args)

    _fix_trainer_args(args=trainer_args)
    _update_trainer_args(args=trainer_args, workspace=workspace)

    trainer_args['callbacks'] = [
        LanguageGeneratorCallback(experiment_workspace=workspace)
    ]

    trainer = Trainer(**trainer_args)

    return trainer


def _fix_trainer_args(args: Dict) -> None:
    args['val_check_interval'] = int(args['val_check_interval'])


def _update_trainer_args(args: Dict, workspace: Workspace) -> None:
    args.update(
        {
            'logger': TensorBoardLogger(
                save_dir=workspace.logs_dir,
                name=workspace.name,
                version=workspace.version
            ),
            'checkpoint_callback': ModelCheckpoint(
                filepath=workspace.models_dir,
                verbose=True,
                save_top_k=2,
                period=0
            )
        }
    )


if __name__ == '__main__':
    main()
