import pathlib
from typing import Optional

import aiohttp
from aiogram import Dispatcher, Bot

from full_stack_transformer.text_generator_telegram_client.handlers import (
    HandlersRegister)


def prepare(
        api_token: str,
        text_generator_service_url: str,
        logs_dir: pathlib.Path,
        text_generator_service_login: Optional[str] = None,
        text_generator_service_password: Optional[str] = None
) -> Dispatcher:
    """Prepares dispatcher object."""

    bot = Bot(token=api_token)
    dispatcher = Dispatcher(bot)

    if text_generator_service_login and text_generator_service_password:
        text_generator_auth = aiohttp.BasicAuth(
            login=text_generator_service_login,
            password=text_generator_service_password)
    else:
        text_generator_auth = None

    handlers_register = HandlersRegister(
        dispatcher=dispatcher,
        text_generator_url=text_generator_service_url,
        text_generator_auth=text_generator_auth,
        logs_dir=logs_dir)

    handlers_register.register_all_handlers()

    return dispatcher
