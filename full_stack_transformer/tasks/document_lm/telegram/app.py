import argparse
import pathlib
from typing import Optional

import aiohttp
from aiogram import Dispatcher, Bot
from aiogram.utils import executor

from full_stack_transformer.tasks.document_lm.telegram.handlers import HandlersRegister

THIS_DIR = pathlib.Path(__file__).parent


def _parse_args():
    logs_dir = THIS_DIR / '../../data/telegram_logs/'

    parser = argparse.ArgumentParser(
        description='This script runs telegram client for the text generator '
                    'service'
    )

    parser.add_argument(
        '--telegram_api_token', type=str, required=True,
        help='Telegram client API token. Could be obtained via `@BotFather` '
             'bot in telegram.'
    )
    parser.add_argument(
        '--text_generator_service_url', type=str, required=True
    )
    parser.add_argument(
        '--text_generator_service_login', type=str, required=False
    )
    parser.add_argument(
        '--text_generator_service_password', type=str, required=False
    )
    parser.add_argument(
        '--logs_dir', type=str, required=False, default=logs_dir
    )

    args = parser.parse_args()
    return args


def main():
    args = _parse_args()

    dispatcher = prepare(
        logs_dir=args.logs_dir,
        telegram_api_token=args.telegram_api_token,
        text_generator_service_url=args.text_generator_service_url,
        text_generator_service_login=args.text_generator_service_login,
        text_generator_service_password=args.text_generator_service_password
    )

    executor.start_polling(dispatcher, skip_updates=True)


def prepare(
        telegram_api_token: str,
        text_generator_service_url: str,
        logs_dir: pathlib.Path,
        text_generator_service_login: Optional[str] = None,
        text_generator_service_password: Optional[str] = None
) -> Dispatcher:
    """Prepares dispatcher object."""

    bot = Bot(token=telegram_api_token)
    dispatcher = Dispatcher(bot)

    if text_generator_service_login and text_generator_service_password:
        text_generator_service_auth = aiohttp.BasicAuth(
            login=text_generator_service_login,
            password=text_generator_service_password
        )
    else:
        text_generator_service_auth = None

    handlers_register = HandlersRegister(
        dispatcher=dispatcher,
        text_generator_service_url=text_generator_service_url,
        text_generator_service_auth=text_generator_service_auth,
        logs_dir=logs_dir
    )

    handlers_register.register_all_handlers()

    return dispatcher


if __name__ == '__main__':
    main()