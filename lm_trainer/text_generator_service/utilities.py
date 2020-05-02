import logging
import pathlib

import torch
from fastapi import FastAPI

from lm_trainer.text_generator_service.views import ViewsRegister
from lm_trainer.pl_modules.model_loading import (
    load_text_generator_from_pl_checkpoint)
from lm_trainer.utilities.log_config import prepare_logging

_LOGGER = logging.getLogger(__name__)


def prepare(checkpoint_path, device, logs_dir) -> FastAPI:
    prepare_logging(pathlib.Path(logs_dir))
    ckpt = torch.load(f=checkpoint_path, map_location='cpu')
    generator = load_text_generator_from_pl_checkpoint(ckpt=ckpt, device=device)
    app = _prepare_app(generator=generator)

    _LOGGER.info('All text_generator_service components successfully initialized.')

    return app


def _prepare_app(generator):
    app = FastAPI()
    views_register = ViewsRegister(app=app, generator=generator)
    views_register.register_all_views()

    return app
