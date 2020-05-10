import pathlib
from multiprocessing import Process, Queue


class TextLinesProducer(Process):
    """Reads input text file lines and sends chunks of them into the queue."""
    def __init__(
            self,
            file_path: pathlib.Path,
            out_queue: Queue,
            chunk_size: int
    ):
        """
        Args:
            file_path:
                Input text file path.

            out_queue:
                Queue to put chunks of lines.

            chunk_size:
                Size of text line chunks to put in the queue. Last chunk, which
                is read from the file could be less than this size.
        """
        super().__init__()

        self._file_path = file_path
        self._out_queue = out_queue
        self._chunk_size = chunk_size

    def run(self) -> None:
        while True:
            chunk = []
            with self._file_path.open() as file:
                for line in file:
                    chunk.append(line)
                    if len(chunk) >= self._chunk_size:
                        self._out_queue.put(chunk)
                        chunk = []

            if len(chunk) > 0:
                self._out_queue.put(chunk)
