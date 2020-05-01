"""Script to train language model.

Pytorch-lightning will add its arguments automatically, so don't forget to
consider them. For example, these ones:

--gradient_clip_val
--gpus
--progress_bar_refresh_rate
--accumulate_grad_batches
--max_epochs
--val_check_interval
--row_log_interval
--precision
--amp_level

Use `python train_model.py --help` to see them all.
"""

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
    callbacks = _prepare_callbacks(
        experiment_dir=experiment_dir,
        tensorboard_logdir=args.tensorboard_logdir)
    _prepare_logging(experiment_dir)

    trainer = _prepare_trainer(args=args, callbacks=callbacks)
    module = LMModule(hparams=args)

    trainer.fit(module)


def _parse_args():
    experiments_root = THIS_DIR / '../../data/experiments'
    tb_logdir = THIS_DIR / '../../data/tb_logs'

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
        '--model_path', type=_str2path, required=True,
        help='Path to the pre-trained GPT model.'
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
        '--tensorboard_logdir', type=_str2path, required=False,
        default=tb_logdir, help='Tensorboard logs directory.'
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
    parser.add_argument(
        '--log_embeddings', required=False, action='store_true',
        help='If this argument set, text embeddings will be logged via the '
             'tensorboard projector.'
    )
    parser.add_argument(
        '--log_text_samples', required=False, action='store_true',
        help='If this argument set, samples which are generated on each '
             'validation epoch end, will be logged to the experiment dir.'
    )

    args = parser.parse_args()

    if args.experiment_name is None:
        args.experiment_name = args.dataset_dir.parents[0].name

    return args


def _str2path(path: str) -> pathlib.Path:
    return pathlib.Path(path)


def _prepare_experiment_dir(args):
    args_dict = args.__dict__.copy()
    if args.resume_from_checkpoint is None or args.experiment_name is not None:
        description = {
            "Experiment": args_dict,
            "Dataset": load_json(args.dataset_dir / 'description.json')}

        experiment_dir = prepare_dataset_dir(
            datasets_root=args.experiments_root,
            dataset_name=args.experiment_name,
            description=description)

        args.description = description
    else:
        experiment_dir = pathlib.Path(args.resume_from_checkpoint).parent

    args.experiment_dir = experiment_dir

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
        {'show_progress_bar': True,
         'val_check_interval': int(trainer_args['val_check_interval'])}
    )
    trainer_args.update(callbacks)

    trainer = pl.Trainer(**trainer_args)

    return trainer


def _prepare_callbacks(
        experiment_dir: pathlib.Path,
        tensorboard_logdir: pathlib.Path
) -> Mapping:
    models_dir = str(experiment_dir / 'models')
    tensorboard_logdir = str(tensorboard_logdir)
    name, version = experiment_dir.parts[-2:]

    callbacks = {
        'logger': TensorBoardLogger(
            save_dir=tensorboard_logdir,
            name=name,
            version=version),
        'checkpoint_callback': ModelCheckpoint(
            filepath=models_dir,
            verbose=True,
            save_top_k=2,
            period=0)
    }

    return callbacks


if __name__ == '__main__':
    main()
