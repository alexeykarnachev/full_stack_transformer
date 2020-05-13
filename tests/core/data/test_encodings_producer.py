from dataclasses import dataclass
from multiprocessing import Queue
from typing import List

import numpy as np

from full_stack_transformer.core.data.encodings_producer import EncodingsProducer
from full_stack_transformer.core.encoding import Encoding


@dataclass
class _FakeTextInput:
    body: str


class _FakeTokenizer:
    def encode_for_train(self, text_input: _FakeTextInput) -> List[Encoding]:
        token_ids = [int(x) for x in list(text_input.body)]
        enc = Encoding(token_ids=token_ids, lm_labels=token_ids)
        return [enc]


def test_encodings_producer():
    inp_text_inputs_queue = Queue()
    out_encodings_queue = Queue()
    text_inputs = []

    for _ in range(200):
        text_input = _FakeTextInput(body=str(np.random.randint(0, 10000)))
        text_inputs.append(text_input)
        inp_text_inputs_queue.put([text_input])

    producer = EncodingsProducer(
        tokenizer=_FakeTokenizer(),
        inp_text_inputs_queue=inp_text_inputs_queue,
        out_encodings_queue=out_encodings_queue
    )

    producer.start()

    for text_input in text_inputs:
        token_ids = [int(x) for x in list(text_input.body)]
        expected_encoding = [Encoding(token_ids=token_ids, lm_labels=token_ids)]
        predicted_encoding = out_encodings_queue.get()
        assert predicted_encoding == expected_encoding
