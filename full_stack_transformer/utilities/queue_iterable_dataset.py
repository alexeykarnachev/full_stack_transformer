import os
from multiprocessing import Queue
from typing import Optional

from torch.utils.data.dataset import IterableDataset


class QueueIterableDataset(IterableDataset):
    def __init__(
            self,
            inp_queue: Queue,
            length: int,
            timeout: Optional[float] = None
    ):
        self._inp_queue = inp_queue
        self._length = length
        self._timeout = timeout

    def __iter__(self):
        while True:
            sample = self._inp_queue.get(timeout=self._timeout)
            yield sample

    def __len__(self):
        return self._length
