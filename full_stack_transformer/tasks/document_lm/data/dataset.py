import pathlib

from full_stack_transformer.core.data.dataset import Dataset
from full_stack_transformer.tasks.document_lm.data.encodings_collate import \
    DocumentEncodingsCollate
from full_stack_transformer.tasks.document_lm.data.text_lines_parsers import \
    DocumentLinesParser
from full_stack_transformer.tasks.document_lm.tokenizer import DocumentTokenizer
from full_stack_transformer.utilities.files import count_lines_in_file


class DocumentDataset(Dataset):
    def __init__(
            self,
            file_path: pathlib.Path,
            tokenizer: DocumentTokenizer,
    ):
        collate = DocumentEncodingsCollate(pad_value=tokenizer.pad_token_id)
        text_lines_parser = DocumentLinesParser()

        super().__init__(
            file_path=file_path,
            tokenizer=tokenizer,
            text_lines_parser=text_lines_parser,
            encodings_collate=collate
        )

    def __len__(self) -> int:
        return count_lines_in_file(self._file_path)
