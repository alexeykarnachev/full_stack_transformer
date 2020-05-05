import argparse

from aiogram.utils import executor

from full_stack_transformer.text_generator_telegram_client.utilities import (
    prepare)


def _parse_args():
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

    args = parser.parse_args()
    return args


def main():
    args = _parse_args()

    dispatcher = prepare(
        api_token=args.telegram_api_token,
        text_generator_service_url=args.text_generator_service_url,
        text_generator_service_login=args.text_generator_service_login,
        text_generator_service_password=args.text_generator_service_password)

    executor.start_polling(dispatcher, skip_updates=True)


if __name__ == '__main__':
    main()
