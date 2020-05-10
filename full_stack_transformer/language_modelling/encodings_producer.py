from multiprocessing import Process, Queue

from full_stack_transformer.language_modelling.tokenization.tokenizer import \
    DocumentTokenizer


class DocumentEncodingsProducer(Process):
    def __init__(
            self,
            tokenizer: DocumentTokenizer,
            inp_text_lines_queue: Queue,
            out_encodings_queue: Queue
    ):
        super().__init__()

        self._tokenizer = tokenizer
        self._inp_queue = inp_text_lines_queue
        self._out_queue = out_encodings_queue

    def run(self) -> None:
        while True:
            lines_chunk = self._inp_queue.get()
            encodings = []
            for line in lines_chunk:
                enc = self._tokenizer.encode_line(line)
                encodings.extend(enc)

            self._out_queue.put(encodings)

