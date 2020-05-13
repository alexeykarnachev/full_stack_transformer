from multiprocessing import Process, Queue


class EncodingsSampler(Process):
    def __init__(self, inp_encodings_queue: Queue, out_encodings_queue: Queue):
        super().__init__(daemon=True)

        self._inp_queue = inp_encodings_queue
        self._out_queue = out_encodings_queue

    def run(self) -> None:
        while True:
            encodings = self._inp_queue.get()
            encodings = sorted(encodings, key=lambda e: -len(e.token_ids))
            for enc in encodings:
                self._out_queue.put(enc)
