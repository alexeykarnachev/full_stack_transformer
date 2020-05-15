from multiprocessing import Queue

import numpy as np
import pytest

from full_stack_transformer.core.data.encodings_sampler import EncodingsSampler
from full_stack_transformer.core.encoding import Encoding


@pytest.mark.parametrize('chunk_size', [1, 2, 10, 500])
def test_encodings_sampler(chunk_size):
    expected_encodings = []

    inp_queue = Queue()
    out_queue = Queue()

    chunk = []

    for i in range(500):
        token_ids = [0] * np.random.randint(0, 100)
        encoding = Encoding(
            token_ids=token_ids,
            lm_labels=[0],
            token_type_ids=None
        )
        chunk.append(encoding)

        if len(chunk) == chunk_size:
            inp_queue.put(chunk)
            sorted_chunk = sorted(chunk, key=lambda x: -len(x.token_ids))
            expected_encodings.append(sorted_chunk)
            chunk = []

    sampler = EncodingsSampler(
        inp_encodings_queue=inp_queue,
        out_encodings_queue=out_queue
    )

    sampler.start()

    while len(expected_encodings):

        expected_chunk = expected_encodings.pop(0)

        while len(expected_chunk):
            predicted_encoding = out_queue.get()
            expected_encoding = expected_chunk.pop(0)

            assert predicted_encoding == expected_encoding
