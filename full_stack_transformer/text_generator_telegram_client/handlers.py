import json
import os
import re
from typing import Optional, Tuple

import aiohttp
from aiogram import Dispatcher, types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

from full_stack_transformer.utilities.strings import get_string_md5


class HandlersRegister:
    def __init__(
            self,
            dispatcher: Dispatcher,
            messages_cache: 'MessagesCache',
            text_generator_url: str,
            text_generator_auth: Optional):
        self._dispatcher = dispatcher
        self._messages_cache = messages_cache
        self._text_generator_url = text_generator_url
        self._text_generator_auth = text_generator_auth

    def register_send_reply_message_handler(self):
        """Replies on user input message."""

        @self._dispatcher.message_handler()
        async def send_reply(message: types.Message):
            message_hash = self._messages_cache.add_message(message.text)

            await self._send_reply(
                message=message,
                seed_string=message.text,
                callback_data=message_hash)

    def register_send_reply_callback_query_handler(self):
        """Replies with user's previous seed text."""

        @self._dispatcher.callback_query_handler(
            lambda q: q.data.startswith('__repeat__:'))
        async def send_reply(callback_query: types.CallbackQuery):
            message_hash = callback_query.data.split(':', 1)[1]
            message_text = self._messages_cache.get_message(message_hash)

            await self._send_reply(
                message=callback_query.message,
                seed_string=message_text,
                callback_data=message_hash)

    async def _send_reply(self, message, seed_string, callback_data):
        response = await self.get_text_generator_service_response(
            seed_string=seed_string)

        reply_text = _prepare_reply_text(
            text_generator_service_response=response,
            prefix_string=seed_string)

        keyboard = _get_inline_keyboard(callback_data=callback_data)

        await message.answer(reply_text, reply_markup=keyboard)

    def register_all_handlers(self):
        for field in dir(self):
            if re.match('^register.+handler$', field):
                getattr(self, field)()

    async def get_text_generator_service_response(
            self,
            seed_string: str) -> Tuple[str, int]:
        url = os.path.join(self._text_generator_url, 'generated_texts')
        payload = {'seed_text': seed_string}
        headers = {'Content-Type': 'application/json'}
        async with aiohttp.ClientSession(
                auth=self._text_generator_auth) as session:
            async with session.post(
                    url=url,
                    data=json.dumps(payload),
                    headers=headers) as response:
                status = response.status
                reply = await response.text()
                return reply, status


def _prepare_reply_text(
        text_generator_service_response: Tuple[str, int],
        prefix_string: str) -> str:
    response_text, status = text_generator_service_response
    if status == 200:
        response_dict = json.loads(response_text)
        generated_text = response_dict['texts'][0]
        generated_text = prefix_string + ' ' + generated_text
    else:
        generated_text = "I'm broken, or tired..."

    return generated_text


def _get_inline_keyboard(callback_data: str) -> InlineKeyboardMarkup:
    buttons = []

    repeat_button = InlineKeyboardButton(
        text='Repeat',
        callback_data=f'__repeat__:{callback_data}')

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
