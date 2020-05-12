import torch.utils.data

from full_stack_transformer.core.data.dataset import Dataset
from full_stack_transformer.core.data.encodings_collate import EncodingsCollate


class DataLoader(torch.utils.data.DataLoader):
    def __init__(
            self,
            dataset: Dataset,
            batch_size: int,
            num_workers: int,
            encodings_collate: EncodingsCollate
    ):
        self._dataset = dataset
        self._batch_size = batch_size
        self._collate = encodings_collate

        super().__init__(
            dataset=self._dataset,
            batch_size=self._batch_size,
            num_workers=num_workers,
            collate_fn=self._collate
        )

    def __len__(self):
        return (len(self._dataset) // self._batch_size) - 1
