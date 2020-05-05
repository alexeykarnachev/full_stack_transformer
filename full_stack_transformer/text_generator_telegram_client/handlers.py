import json
import logging
import os
import pathlib
from http import HTTPStatus
from typing import Optional, Tuple

import aiohttp
from aiogram import Dispatcher, types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, ParseMode
from aiohttp import ServerDisconnectedError

from full_stack_transformer.utilities.strings import get_string_md5


class HandlersRegister:
    """Class which registers all text generator telegram client handlers."""

    WELCOME_MESSAGE = "Hello, send me message and I'll continue it."
    TEXT_GENERATOR_SERVICE_ERROR_MESSAGE = "I'm broken, or tired..."
    CANT_REPEAT_MESSAGE = "Can't repeat, please send me a new one."

    REPEAT_CALLBACK_DATA_PREFIX = '__repeat__'

    def __init__(
            self,
            dispatcher: Dispatcher,
            text_generator_service_url: str,
            text_generator_service_auth: Optional,
            logs_dir: pathlib.Path
    ):
        self._dispatcher = dispatcher
        self._messages_cache = MessagesCache()
        self._text_generator_service_url = text_generator_service_url
        self._text_generator_service_auth = text_generator_service_auth
        self._logging_handler = LoggingHandler(logs_dir=logs_dir)

    def register_start_message_handler(self):
        """Handles `/start` command and sends welcome message."""

        @self._dispatcher.message_handler(commands=['start'])
        async def start(message: types.Message):
            await message.answer(self.WELCOME_MESSAGE)

    def register_send_reply_message_handler(self):
        """Replies on user input message."""

        @self._dispatcher.message_handler()
        async def send_reply(message: types.Message):
            message_hash = self._messages_cache.add_message(message.text)

            await self._get_response_and_send_reply(
                message=message,
                seed_string=message.text,
                callback_data=message_hash)

    def register_send_reply_callback_query_handler(self):
        """Replies with user's previous seed text."""

        @self._dispatcher.callback_query_handler(
            lambda q: q.data.startswith(self.REPEAT_CALLBACK_DATA_PREFIX)
        )
        async def send_reply(callback_query: types.CallbackQuery):
            message_hash = callback_query.data.split(':', 1)[1]
            message_text = self._messages_cache.get_message(message_hash)

            if message_text is None:
                await callback_query.message.answer(self.CANT_REPEAT_MESSAGE)
            else:
                await self._get_response_and_send_reply(
                    message=callback_query.message,
                    seed_string=message_text,
                    callback_data=message_hash)

    async def _get_response_and_send_reply(
            self,
            message,
            seed_string,
            callback_data):
        response = await self._get_text_generator_service_response(
            seed_string=seed_string)

        reply_text = _prepare_reply_text(
            text_generator_service_response=response,
            prefix_string=seed_string)

        keyboard = _get_inline_keyboard(callback_data=callback_data)

        self._logging_handler.log(
            user_id=_get_user_id_from_message(message=message),
            log_msg=f'\n{reply_text}\n')

        await message.answer(
            text=reply_text,
            reply_markup=keyboard,
            parse_mode=ParseMode.MARKDOWN)

    async def _get_text_generator_service_response(
            self,
            seed_string: str) -> Tuple[Optional[str], int]:
        url = os.path.join(self._text_generator_service_url, 'generated_texts')
        payload = {'seed_text': seed_string}
        headers = {'Content-Type': 'application/json'}
        async with aiohttp.ClientSession(
                auth=self._text_generator_service_auth) as session:
            try:
                async with session.post(
                        url=url,
                        data=json.dumps(payload),
                        headers=headers) as response:
                    status = response.status
                    reply = await response.text()
                    return reply, status
            except ServerDisconnectedError:
                return None, HTTPStatus.INTERNAL_SERVER_ERROR

    def register_all_handlers(self):
        self.register_start_message_handler()
        self.register_send_reply_message_handler()
        self.register_send_reply_callback_query_handler()


def _prepare_reply_text(
        text_generator_service_response: Tuple[Optional[str], int],
        prefix_string: str) -> str:
    response_text, status = text_generator_service_response
    if status == 200:
        response_dict = json.loads(response_text)
        generated_text = response_dict['texts'][0]
        generated_text = f'*{prefix_string}* {generated_text}'
    else:
        generated_text = HandlersRegister.TEXT_GENERATOR_SERVICE_ERROR_MESSAGE

    return generated_text


def _get_user_id_from_message(message):
    username = str(message.chat['username'])
    chat_id = str(message.chat['id'])
    user_name = "".join(x for x in username if x.isalnum())
    user_id = user_name + '_' + chat_id
    return user_id


def _get_inline_keyboard(callback_data: str) -> InlineKeyboardMarkup:
    buttons = []
    repeat_button = InlineKeyboardButton(
        text='Repeat',
        callback_data=f'{HandlersRegister.REPEAT_CALLBACK_DATA_PREFIX}:'
                      f'{callback_data}')

    buttons.append(repeat_button)

    keyboard = InlineKeyboardMarkup(inline_keyboard=[buttons])

    return keyboard


class MessagesCache:
    def __init__(self):
        self._cache = dict()

    def add_message(self, message: str) -> str:
        message_hash = get_string_md5(message)[:16]
        self._cache[message_hash] = message
        return message_hash

    def get_message(self, message_hash: str) -> str:
        message = self._cache.get(message_hash)
        return message


class LoggingHandler:
    """Performs logging for users in their individual files."""

    FORMATTER = '%(asctime)s %(message)s'

    def __init__(self, logs_dir: pathlib.Path):
        self._logs_dir = logs_dir
        self._cache = dict()

        self._logs_dir.mkdir(parents=True, exist_ok=True)

    def _get_logger(self, user_id: str):
        if user_id not in self._cache:
            log_file = self._logs_dir / f'{user_id}.log'
            formatter = logging.Formatter(self.FORMATTER)
            handler = logging.FileHandler(str(log_file))
            handler.setFormatter(formatter)

            logger = logging.getLogger(f'{user_id}')
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)
        else:
            logger = self._cache[user_id]

        return logger

    def log(self, user_id: str, log_msg: str):
        logger = self._get_logger(user_id=user_id)
        logger.info(log_msg)
