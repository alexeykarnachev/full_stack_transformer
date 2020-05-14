import logging
import pathlib

import torch
from fastapi import FastAPI

import full_stack_transformer
from full_stack_transformer.tasks.common.language_generator.generator import LanguageGenerator
from full_stack_transformer.tasks.common.models.hf_gpt2 import load_model_from_checkpoint
from full_stack_transformer.tasks.document_decoder import load_tokenizer_from_checkpoint
from full_stack_transformer.tasks.document_decoder.serving.views import ViewsRegister
from full_stack_transformer.utilities.log_config import prepare_logging

_LOGGER = logging.getLogger(__name__)


def prepare(checkpoint_path, device, logs_dir) -> FastAPI:
    prepare_logging(pathlib.Path(logs_dir))
    ckpt = torch.load(f=checkpoint_path, map_location='cpu')

    tokenizer = load_tokenizer_from_checkpoint(ckpt=ckpt)

    model = load_model_from_checkpoint(ckpt=ckpt, device=device)
    generator = LanguageGenerator(
        model=model,
        eos_token_id=tokenizer.eos_token_id
    )
    version = _get_version_from_ckpt(
        ckpt=ckpt,
        checkpoint_path=checkpoint_path
    )
    app = _prepare_app(
        generator=generator,
        tokenizer=tokenizer,
        version=version
    )

    _LOGGER.info(
        'All text_generator_service components were successfully initialized.'
    )

    return app


def _get_version_from_ckpt(ckpt, **kwargs):
    version = dict()

    version['package_version'] = full_stack_transformer.__version__
    version['epoch'] = ckpt['epoch']
    version['global_step'] = ckpt['global_step']
    version['hparams'] = ckpt['hparams']
    version.update(kwargs)

    return version


def _prepare_app(generator, tokenizer, version):
    app = FastAPI()
    views_register = ViewsRegister(
        app=app,
        generator=generator,
        version=version,
        tokenizer=tokenizer
    )
    views_register.register_all_views()

    return app
