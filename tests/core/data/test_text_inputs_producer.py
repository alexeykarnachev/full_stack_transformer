import pathlib
from dataclasses import dataclass
from typing import Optional

import pytest

from full_stack_transformer.core.data.text_inputs_producer import TextInputsProducer
from full_stack_transformer.core.data.text_lines_parsers import TextLinesParser
from full_stack_transformer.core.text_input import TextInput
from tests.utilities import generate_random_text_file

from multiprocessing import Queue


@dataclass
class _FakeTextInput(TextInput):
    body: str


class _FakeTextLinesParser(TextLinesParser):
    def __init__(self):
        pass

    def parse(self, text: str) -> Optional[_FakeTextInput]:
        return _FakeTextInput(body=text)


@pytest.mark.parametrize('chunk_size', [1, 3, 10, 1000, 5000])
def test_text_inputs_producer(chunk_size, tmp_path):
    file_path = tmp_path / 'random.txt'

    generate_random_text_file(file_path, 1000)

    with pathlib.Path(file_path).open() as file:
        expected_lines = file.readlines()

    out_queue = Queue()

    producer = TextInputsProducer(
        file_path=file_path,
        out_text_inputs_queue=out_queue,
        out_chunk_size=chunk_size,
        text_lines_parser=_FakeTextLinesParser()
    )

    producer.start()

    lines = []
    successes = 0
    while True:
        chunk = out_queue.get()

        lines.extend([x.body for x in chunk])

        if len(lines) >= len(expected_lines):
            predicted_lines = lines[:len(expected_lines)]
            lines = lines[len(expected_lines):]

            assert predicted_lines == expected_lines
            successes += 1

            if successes == 5:
                break
