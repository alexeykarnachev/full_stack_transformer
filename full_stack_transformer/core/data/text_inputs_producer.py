import pathlib
from multiprocessing import Process, Queue

from full_stack_transformer.core.data.text_lines_parsers import TextLinesParser


class TextInputsProducer(Process):
    def __init__(
            self,
            file_path: pathlib.Path,
            out_text_inputs_queue: Queue,
            out_chunk_size: int,
            text_lines_parser: TextLinesParser
    ):
        super().__init__(daemon=True)

        self._file_path = file_path
        self._chunk_size = out_chunk_size
        self._out_queue = out_text_inputs_queue
        self._parser = text_lines_parser

    def run(self) -> None:
        while True:
            chunk = []
            with self._file_path.open() as file:
                for line in file:
                    text_inputs = self._parser.parse(line)

                    chunk.extend(text_inputs)

                    if len(chunk) >= self._chunk_size:
                        self._out_queue.put(chunk)
                        chunk = []

            if len(chunk) > 0:
                self._out_queue.put(chunk)
