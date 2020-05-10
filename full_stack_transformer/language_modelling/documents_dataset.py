import pathlib
from multiprocessing import Queue
from typing import Optional

from torch.utils.data.dataloader import DataLoader

from full_stack_transformer.core.files import count_lines_in_file
from full_stack_transformer.core.queue_iterable_dataset import \
    QueueIterableDataset
from full_stack_transformer.core.text_lines_producer import \
    TextLinesProducer
from full_stack_transformer.language_modelling.encodings_collate import \
    DocumentEncodingsCollate
from full_stack_transformer.language_modelling.encodings_producer import \
    DocumentEncodingsProducer
from full_stack_transformer.language_modelling.encodings_sampler import \
    DocumentEncodingsSampler
from full_stack_transformer.language_modelling.tokenization.tokenizer import \
    DocumentTokenizer

_QUEUE_MAX_SIZE = 10


class DocumentDataset(QueueIterableDataset):
    def __init__(
            self,
            file_path: pathlib.Path,
            tokenizer: DocumentTokenizer,
            n_producer_workers: int = 4,
            chunk_size: Optional[int] = 10000
    ):
        """
        Args:
            file_path:
                Path to the document samples file.

            tokenizer:
                Document tokenizer object.

            n_producer_workers:
                Number of encodings producer workers.

            chunk_size:
                Chunk size to operate on.
        """
        self._file_path = file_path
        self._tokenizer = tokenizer
        self._n_producer_workers = n_producer_workers
        self._chunk_size = chunk_size

        sorted_samples_queue, length = self._initialize()

        super().__init__(
            inp_queue=sorted_samples_queue,
            length=length
        )

    def _initialize(self):
        lines_queue = Queue(maxsize=_QUEUE_MAX_SIZE)
        encodings_queue = Queue(maxsize=_QUEUE_MAX_SIZE)
        sorted_encodings_queue = Queue(maxsize=_QUEUE_MAX_SIZE)

        length = count_lines_in_file(self._file_path)

        lines_producer = TextLinesProducer(
            file_path=self._file_path,
            out_queue=lines_queue,
            chunk_size=self._chunk_size
        )

        model_input_producers = []

        for _ in range(self._n_producer_workers):
            samples_producer = DocumentEncodingsProducer(
                inp_text_lines_queue=lines_queue,
                out_encodings_queue=encodings_queue,
                tokenizer=self._tokenizer
            )
            model_input_producers.append(samples_producer)

        samples_sampler = DocumentEncodingsSampler(
            inp_encodings_queue=encodings_queue,
            out_encodings_queue=sorted_encodings_queue
        )

        lines_producer.start()
        [p.start() for p in model_input_producers]
        samples_sampler.start()

        return sorted_encodings_queue, length

    def get_data_loader(
            self,
            batch_size: int,
            pad_value: int,
            num_workers: int = 0
    ) -> DataLoader:
        dataloader = DocumentsDataLoader(
            dataset=self,
            batch_size=batch_size,
            pad_value=pad_value,
            num_workers=num_workers
        )

        return dataloader


class DocumentsDataLoader(DataLoader):
    def __init__(
            self,
            dataset: DocumentDataset,
            batch_size: int,
            num_workers: int,
            pad_value: int
    ):
        self._dataset = dataset
        self._batch_size = batch_size

        collate_fn = DocumentEncodingsCollate(
            pad_value=pad_value
        )

        super().__init__(
            dataset=self._dataset,
            batch_size=self._batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn
        )

    def __len__(self):
        return (len(self._dataset) // self._batch_size) - 1
