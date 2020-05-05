import argparse
import pathlib

from aiogram.utils import executor

from full_stack_transformer.scripts.utilities import str2path
from full_stack_transformer.text_generator_telegram_client.utilities import (
    prepare)

THIS_DIR = pathlib.Path(__file__).parent


def _parse_args():
    logs_dir = THIS_DIR / '../../data/text_generator_telegram_client_logs'

    parser = argparse.ArgumentParser(
        description='This script runs telegram client for the text generator '
                    'service')

    parser.add_argument(
        '--telegram_api_token', type=str, required=True,
        help='Telegram client API token. Could be obtained via `@BotFather` '
             'bot in telegram.')
    parser.add_argument(
        '--text_generator_service_url', type=str, required=True)
    parser.add_argument(
        '--text_generator_service_login', type=str, required=False)
    parser.add_argument(
        '--text_generator_service_password', type=str, required=False)
    parser.add_argument(
        '--logs_dir', type=str2path, required=False, default=logs_dir)

    args = parser.parse_args()
    return args


def main():
    args = _parse_args()

    dispatcher = prepare(
        logs_dir=args.logs_dir,
        api_token=args.telegram_api_token,
        text_generator_service_url=args.text_generator_service_url,
        text_generator_service_login=args.text_generator_service_login,
        text_generator_service_password=args.text_generator_service_password)

    executor.start_polling(dispatcher, skip_updates=True)


if __name__ == '__main__':
    main()
