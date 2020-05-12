import abc
import pathlib
from multiprocessing import Queue
from typing import Optional, Callable

from full_stack_transformer.core.data.dataloader import DataLoader
from full_stack_transformer.core.data.encodings_producer import EncodingsProducer
from full_stack_transformer.core.data.encodings_sampler import EncodingsSampler
from full_stack_transformer.core.data.text_inputs_producer import TextInputsProducer
from full_stack_transformer.core.data.text_lines_parsers import TextLinesParser
from full_stack_transformer.core.tokenizer import Tokenizer
from full_stack_transformer.utilities.queue_iterable_dataset import \
    QueueIterableDataset

_QUEUE_MAX_SIZE = 10


class Dataset(QueueIterableDataset):
    def __init__(
            self,
            file_path: pathlib.Path,
            tokenizer: Tokenizer,
            text_lines_parser: TextLinesParser,
            encodings_collate: Callable,
            n_producer_workers: int = 4,
            chunk_size: Optional[int] = 10000
    ):
        self._file_path = file_path
        self._tokenizer = tokenizer
        self._n_producer_workers = n_producer_workers
        self._chunk_size = chunk_size
        self._text_lines_parser = text_lines_parser
        self._collate = encodings_collate

        sorted_samples_queue, length = self._initialize()

        super().__init__(
            inp_queue=sorted_samples_queue,
            length=length
        )

    def _initialize(self):
        text_inputs_queue = Queue(maxsize=_QUEUE_MAX_SIZE)
        encodings_queue = Queue(maxsize=_QUEUE_MAX_SIZE)
        sorted_encodings_queue = Queue(maxsize=_QUEUE_MAX_SIZE)

        text_inputs_producer = TextInputsProducer(
            file_path=self._file_path,
            out_text_inputs_queue=text_inputs_queue,
            out_chunk_size=self._chunk_size,
            text_lines_parser=self._text_lines_parser
        )

        model_inputs_producers = []

        for _ in range(self._n_producer_workers):
            samples_producer = EncodingsProducer(
                inp_text_inputs_queue=text_inputs_queue,
                out_encodings_queue=encodings_queue,
                tokenizer=self._tokenizer
            )
            model_inputs_producers.append(samples_producer)

        encodings_sampler = EncodingsSampler(
            inp_encodings_queue=encodings_queue,
            out_encodings_queue=sorted_encodings_queue
        )

        text_inputs_producer.start()
        [p.start() for p in model_inputs_producers]
        encodings_sampler.start()

        return sorted_encodings_queue

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    def get_data_loader(
            self,
            batch_size: int,
            num_workers: int
    ) -> DataLoader:
        dl = DataLoader(
            dataset=self,
            batch_size=batch_size,
            num_workers=num_workers,
            encodings_collate=self._collate
        )
        return dl
