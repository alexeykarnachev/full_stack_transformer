import torch
from fastapi import FastAPI

from lm_trainer.application.views import ViewsRegister
from lm_trainer.pl_modules.model_loading import (
    load_transformer_model_from_pl_checkpoint,
    load_tokenizer_from_checkpoint
)
from lm_trainer.text_generator.text_generator import TextGenerator


def prepare(checkpoint_path, device) -> FastAPI:
    ckpt = torch.load(f=checkpoint_path, map_location=device)
    generator = _prepare_generator(ckpt=ckpt)
    app = _prepare_app(generator=generator)

    return app


def _prepare_generator(ckpt):
    model = load_transformer_model_from_pl_checkpoint(ckpt=ckpt)
    tokenizer = load_tokenizer_from_checkpoint(ckpt=ckpt)
    generator = TextGenerator(model=model, tokenizer=tokenizer)

    return generator


def _prepare_app(generator):
    app = FastAPI()
    views_register = ViewsRegister(app=app, generator=generator)
    views_register.register_all_views()

    return app
