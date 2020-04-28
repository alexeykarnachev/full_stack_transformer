import pathlib
import re
from typing import Sequence, Callable, Generator

import more_itertools
import numpy as np
import torch.utils.data
import tqdm
from torch.utils.data.dataloader import DataLoader

from lm_trainer.datasets.length_sort_sampler import LengthSortSampler
from lm_trainer.tokenizers import Tokenizer
from lm_trainer.utilities.sequences import pad_sequences


class DocumentsDatasetError(Exception):
    pass


class DocumentsDataset(torch.utils.data.Dataset):
    CORPUS_FILE = 'corpus.npy'
    SAMPLE_START_POSITIONS_FILE = 'sample_start_positions.npy'

    def __init__(
            self,
            corpus: np.array,
            sample_start_positions: np.array
    ):
        self._corpus = corpus
        self._sample_start_positions = sample_start_positions

    def __len__(self):
        return len(self._sample_start_positions)

    def __getitem__(self, index):
        sample_start_position = self._sample_start_positions[index]
        sample_end_position = self._get_sample_end_position(index)

        document = self._corpus[sample_start_position:sample_end_position]

        return document

    def _get_sample_end_position(self, index):
        try:
            sample_end_position = self._sample_start_positions[index + 1]
        except IndexError:
            sample_end_position = len(self._corpus)

        return sample_end_position

    def save(self, dir_path: pathlib.Path) -> None:
        np.save(
            str(dir_path / self.CORPUS_FILE),
            self._corpus)
        np.save(
            str(dir_path / self.SAMPLE_START_POSITIONS_FILE),
            self._sample_start_positions)

    def _get_sample_lengths(self):
        sample_lengths = np.diff(
            a=self._sample_start_positions,
            append=len(self._corpus))
        return sample_lengths

    def get_dataloader(self, batch_size: int, pad_val: int) -> DataLoader:

        collate_fn = DocumentsDatasetCollate(pad_val=pad_val)
        sampler = LengthSortSampler(
            sample_lengths=self._get_sample_lengths(),
            batch_size=batch_size)

        dataloader = DataLoader(
            dataset=self,
            batch_size=batch_size,
            collate_fn=collate_fn,
            sampler=sampler,
            drop_last=False)

        return dataloader


def load_from_dir(dir_path: pathlib.Path) -> DocumentsDataset:
    corpus = np.load(
        str(dir_path / DocumentsDataset.CORPUS_FILE))
    sample_start_positions = np.load(
        str(dir_path / DocumentsDataset.SAMPLE_START_POSITIONS_FILE))

    dataset = DocumentsDataset(
        corpus=corpus,
        sample_start_positions=sample_start_positions)
    return dataset


class DocumentsDatasetReader:
    def __init__(
            self,
            file_path: pathlib.Path,
            end_of_document: str,
            document_text_preprocessor: Callable[[str], str],
            tokenizer: Tokenizer,
            max_sample_length: int,
            min_sample_length: int,
            chunk_size: int = 10000
    ):
        self._file_path = file_path
        self._end_of_document = end_of_document
        self._document_text_preprocessor = document_text_preprocessor
        self._chunk_size = chunk_size
        self._tokenizer = tokenizer
        self._max_sample_length = max_sample_length
        self._min_sample_length = min_sample_length

        self._check_arguments_validity()

    def construct(self) -> DocumentsDataset:
        corpus = []
        sample_start_pos = []

        for sample_token_ids in self._iterate_on_sample_token_ids():
            if len(sample_token_ids) < self._min_sample_length:
                continue

            sample_start_pos.append(len(corpus))
            corpus.extend(sample_token_ids)

        corpus = np.array(corpus, dtype=np.int32)
        sample_start_pos = np.array(sample_start_pos, dtype=np.int32)

        dataset = DocumentsDataset(
            corpus=corpus,
            sample_start_positions=sample_start_pos)

        return dataset

    def _iterate_on_sample_token_ids(
            self) -> Generator[Sequence[int], None, None]:

        for document_token_ids in self._iterate_on_document_token_ids():
            for sample_token_ids in more_itertools.chunked(
                    document_token_ids, self._max_sample_length):
                yield sample_token_ids

    def _iterate_on_document_token_ids(
            self) -> Generator[Sequence[int], None, None]:
        documents_iterator = self._iterate_on_documents()

        for documents_chunk in more_itertools.chunked(
                documents_iterator, n=self._chunk_size
        ):
            documents_chunk = self._preprocess_documents(documents_chunk)
            encodings_chunk = self._tokenizer.encode_batch(
                sequences=documents_chunk)

            for encoding in encodings_chunk:
                token_ids = encoding.ids
                yield token_ids

    def _preprocess_documents(self, documents: Sequence[str]):
        preprocessed_documents = []
        for document in documents:
            document = self._document_text_preprocessor(document)
            preprocessed_documents.append(document)

        return preprocessed_documents

    def _iterate_on_documents(self) -> Generator[str, None, None]:
        document_lines = []

        with self._file_path.open() as file:

            progress_bar = tqdm.tqdm(
                iterable=file,
                desc='Constructing datasets')

            for line in progress_bar:
                striped_line = line.strip()
                if striped_line != self._end_of_document:
                    document_lines.append(striped_line)
                else:
                    document = '\n'.join(document_lines)
                    document_lines = []

                    yield document

    def _check_arguments_validity(self):
        striped_end_of_document = self._end_of_document.strip()

        if len(striped_end_of_document) == 0:
            raise DocumentsDatasetError(
                'Striped `end_of_document` length must be > 0.')
        elif re.search(r'\s+', striped_end_of_document):
            raise DocumentsDatasetError(
                'Striped `end_of_document` must not contain white space '
                'symbols.')


class DocumentsDatasetCollate:
    def __init__(self, pad_val: int):
        self._pad_val = pad_val

    def __call__(self, documents_batch):
        documents_batch = pad_sequences(documents_batch, pad_val=self._pad_val)
        return torch.tensor(documents_batch, dtype=torch.long)
