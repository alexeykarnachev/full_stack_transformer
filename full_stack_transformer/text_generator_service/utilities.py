import logging
import pathlib

import torch
from fastapi import FastAPI

import full_stack_transformer
from full_stack_transformer.pl_modules.model_loading import \
    load_text_generator_from_pl_checkpoint
from full_stack_transformer.text_generator_service.views import ViewsRegister
from full_stack_transformer.utilities.log_config import prepare_logging

_LOGGER = logging.getLogger(__name__)


def prepare(checkpoint_path, device, logs_dir) -> FastAPI:
    prepare_logging(pathlib.Path(logs_dir))
    ckpt = torch.load(f=checkpoint_path, map_location='cpu')
    generator = load_text_generator_from_pl_checkpoint(ckpt=ckpt, device=device)
    version = _get_version_from_ckpt(ckpt, checkpoint_path)
    app = _prepare_app(generator=generator, version=version)

    _LOGGER.info(
        'All text_generator_service components were successfully initialized.')

    return app


def _get_version_from_ckpt(ckpt, checkpoint_path):
    version = dict()

    version['package_version'] = full_stack_transformer.__version__
    version['epoch'] = ckpt['epoch']
    version['global_step'] = ckpt['global_step']
    version['checkpoint_path'] = checkpoint_path
    version['description'] = ckpt['hparams']['description']

    return version


def _prepare_app(generator, version):
    app = FastAPI()
    views_register = ViewsRegister(
        app=app,
        generator=generator,
        version=version)
    views_register.register_all_views()

    return app
