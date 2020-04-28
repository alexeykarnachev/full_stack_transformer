import argparse
import copy
import logging.config
import pathlib
from typing import Mapping

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from lm_trainer.pl_modules.lm_module import LMModule
from lm_trainer.utilities.file_io import prepare_dataset_dir, load_json
from lm_trainer.utilities.log_config import get_log_config
from lm_trainer.utilities.training import seed_everything

THIS_DIR = pathlib.Path(__file__).parent


def main():
    args = _parse_args()
    seed_everything(seed=args.seed)

    experiment_dir = _prepare_experiment_dir(args)
    callbacks = _prepare_callbacks(experiment_dir=experiment_dir)
    _prepare_logging(experiment_dir)

    trainer = _prepare_trainer(args=args, callbacks=callbacks)
    module = LMModule(hparams=args)

    trainer.fit(module)


def _parse_args():
    experiments_root = THIS_DIR / '../../data/experiments'

    parser = argparse.ArgumentParser(
        description='Script which executes lm-model training.')

    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument(
        '--dataset_dir', type=_str2path, required=True,
        help='Path to the dataset directory. It must contain train, valid sub '
             'folders and description.json file. Such a dir is prepared by '
             '`prepare_dataset.py` script.'
    )
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path or name of a pre-trained GPT model.'
    )
    parser.add_argument(
        '--batch_size', type=int, required=True, help='Batch size'
    )
    parser.add_argument(
        '--learning_rate', type=float, required=True, help='Learning rate.'
    )
    parser.add_argument(
        '--num_warmup_steps', type=int, required=True,
        help='Number of warmup steps for lr-scheduler.'
    )
    parser.add_argument(
        '--num_cycles', type=int, required=True,
        help='Number of cosine lr-scheduler cycles.'
    )
    parser.add_argument(
        '--seed', type=int, required=False, default=228, help='Random seed.'
    )
    parser.add_argument(
        '--experiments_root', type=_str2path, required=False,
        default=experiments_root,
        help='Path to the high level experiments root directory, where all '
             'your specific experiments subdirectories are.'
    )
    parser.add_argument(
        '--experiment_name', type=str, required=False, default=None,
        help='Experiment base name. The experiment will be saved in the '
             'directory: experiments_root/experiment_name/experiment_version. '
             'If None (default), the dataset name will be assumed to be the '
             'experiment name.'
    )

    args = parser.parse_args()
    return args


def _str2path(path: str) -> pathlib.Path:
    return pathlib.Path(path)


def _prepare_experiment_dir(args):
    args_dict = args.__dict__.copy()
    if args.resume_from_checkpoint is None or args.experiment_name is not None:
        description = {
            "Experiment": args_dict,
            "Dataset": load_json(args.dataset_dir / 'description.json'),
            "GPT": load_json(args.gpt_dir / 'config.json')}

        experiment_dir = prepare_dataset_dir(
            datasets_root=args.experiments_root,
            dataset_name=args.experiment_name,
            description=description)
    else:
        experiment_dir = pathlib.Path(args.resume_from_checkpoint).parent

    return experiment_dir


def _prepare_logging(experiment_dir):
    """Configures logging."""
    log_config = get_log_config(experiment_dir / 'logs')
    logging.config.dictConfig(log_config)


def _prepare_trainer(
        args: argparse.Namespace,
        callbacks: Mapping
) -> pl.Trainer:
    trainer_args = copy.deepcopy(args.__dict__)

    trainer_args.update(
        {
            'show_progress_bar': True,
            'progress_bar_refresh_rate': 50,
            'row_log_interval': 50,
            'val_check_interval': int(trainer_args['val_check_interval'])
        }
    )
    trainer_args.update(callbacks)

    trainer = pl.Trainer(**trainer_args)

    return trainer


def _prepare_callbacks(experiment_dir: pathlib.Path) -> Mapping:
    models_dir = str(experiment_dir / 'models')
    tensorboard_logdir = str(experiment_dir / 'tb_logs')

    callbacks = {
        'tb_logger_callback': TensorBoardLogger(save_dir=tensorboard_logdir),
        'model_checkpoint_callback': ModelCheckpoint(
            filepath=models_dir,
            verbose=True,
            save_top_k=2,
            period=0)
    }

    return callbacks


if __name__ == '__main__':
    main()
