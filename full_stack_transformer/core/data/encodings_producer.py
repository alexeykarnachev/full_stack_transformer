from multiprocessing import Process, Queue

from full_stack_transformer.core.tokenizer import Tokenizer


class EncodingsProducer(Process):
    def __init__(
            self,
            tokenizer: Tokenizer,
            inp_text_inputs_queue: Queue,
            out_encodings_queue: Queue
    ):
        super().__init__()

        self._tokenizer = tokenizer
        self._inp_queue = inp_text_inputs_queue
        self._out_queue = out_encodings_queue

    def run(self) -> None:
        while True:
            input_chunk = self._inp_queue.get()
            encodings = []
            for text_input in input_chunk:
                enc = self._tokenizer.encode_for_train(text_input)
                encodings.extend(enc)

            self._out_queue.put(encodings)
