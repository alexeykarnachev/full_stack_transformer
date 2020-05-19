import pathlib

from full_stack_transformer.core.data.dataset import Dataset
from full_stack_transformer.core.data.encodings_collate import EncodingsCollate
from full_stack_transformer.tasks.document_decoder.data.text_lines_parser import JsonLinesParser
from full_stack_transformer.tasks.document_decoder.text_input import DocumentInput
from full_stack_transformer.tasks.document_decoder.tokenizer import DocumentTokenizer
from full_stack_transformer.utilities.files import count_lines_in_file


class DocumentDataset(Dataset):
    def __init__(
            self,
            file_path: pathlib.Path,
            tokenizer: DocumentTokenizer,
    ):
        collate = EncodingsCollate(pad_value=tokenizer.pad_token_id)
        text_lines_parser = JsonLinesParser(text_input_cls=DocumentInput)

        super().__init__(
            file_path=file_path,
            tokenizer=tokenizer,
            text_lines_parser=text_lines_parser,
            encodings_collate=collate
        )

    def __len__(self) -> int:
        return count_lines_in_file(self._file_path)
