import abc

import tokenizers


class Tokenizer(abc.ABC, tokenizers.SentencePieceBPETokenizer):
    @abc.abstractmethod
    def get_pad_token(self) -> str:
        pass

    @abc.abstractmethod
    def get_eos_token(self) -> str:
        pass

    @abc.abstractmethod
    def get_bos_token(self) -> str:
        pass

    def get_pad_token_id(self) -> int:
        return self.token_to_id(self.get_pad_token())

    def get_eos_token_id(self) -> int:
        return self.token_to_id(self.get_eos_token())

    def get_bos_token_id(self) -> int:
        return self.token_to_id(self.get_bos_token())

    @abc.abstractmethod
    def preprocess(self, sequence: str) -> str:
        pass

    @abc.abstractmethod
    def postprocess(self, sequence: str) -> str:
        pass

    def preprocess_document(self, sequence: str) -> str:
        sequence = self.preprocess(sequence)
        sequence = f'{self.get_bos_token()} {sequence} {self.get_eos_token()}'
        return sequence
