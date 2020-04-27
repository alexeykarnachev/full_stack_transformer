import math
import pathlib

import numpy as np

import lm_trainer.tokenizers.ru_transformers_tokenizer as ru_transformer
from lm_trainer.datasets.documents_dataset import (
    DocumentsDatasetReader, load_from_dir, DocumentsDataset
)

_DATA_DIR = pathlib.Path(__file__).parent / 'data'
_DOCUMENTS_FILE = _DATA_DIR / 'documents.txt'


def test_documents_dataset(tmp_path):
    """Documents dataset reading, construction characterization test."""
    preprocessor = ru_transformer.preprocessor
    tokenizer = ru_transformer.RuTransformersTokenizer()

    max_sample_length = 36
    min_sample_length = 16

    reader = DocumentsDatasetReader(
        file_path=_DOCUMENTS_FILE,
        end_of_document='|',
        document_text_preprocessor=preprocessor,
        tokenizer=tokenizer,
        max_sample_length=max_sample_length,
        min_sample_length=min_sample_length)

    dataset = reader.construct()

    assert len(dataset) == 1164

    for i in range(len(dataset)):
        sample = dataset[i]
        assert min_sample_length <= len(sample) <= max_sample_length

    dataset.save(tmp_path)

    dataset_loaded = load_from_dir(tmp_path)

    assert np.array_equal(
        dataset._corpus,
        dataset_loaded._corpus)

    assert np.array_equal(
        dataset._sample_start_positions,
        dataset_loaded._sample_start_positions)


def test_documents_dataloader():
    batch_size = 7

    dataset = DocumentsDataset(
        corpus=np.arange(0, 1000),
        sample_start_positions=np.arange(0, 990, 10))

    dataloader = dataset.get_dataloader(
        batch_size=batch_size,
        pad_val=0)

    assert len(dataloader) == math.ceil(len(dataset) / batch_size)

    for i_batch, documents_batch in enumerate(dataloader):

        n_samples = documents_batch.size()[0]

        if i_batch == len(dataloader) - 1:
            assert n_samples == len(dataset) % batch_size
        else:
            assert n_samples == batch_size
