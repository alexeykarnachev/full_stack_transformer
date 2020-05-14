import json
import logging
import pathlib

from full_stack_transformer.core.data.dataset import Dataset
from full_stack_transformer.core.data.encodings_collate import EncodingsCollate
from full_stack_transformer.tasks.common.line_parsers.json_lines import JsonLinesParser
from full_stack_transformer.tasks.common.text_inputs.dialog import DialogInput
from full_stack_transformer.tasks.common.tokenizers.dialog import DialogTokenizer

_LOGGER = logging.getLogger(__name__)


class DialogDataset(Dataset):
    def __init__(
            self,
            file_path: pathlib.Path,
            tokenizer: DialogTokenizer,
    ):
        collate = EncodingsCollate(pad_value=tokenizer.pad_token_id)
        text_lines_parser = JsonLinesParser(text_input_cls=DialogInput)

        super().__init__(
            file_path=file_path,
            tokenizer=tokenizer,
            text_lines_parser=text_lines_parser,
            encodings_collate=collate
        )

    def __len__(self) -> int:
        return _count_samples_in_file(self._file_path)


def _count_samples_in_file(file_path: pathlib.Path) -> int:
    n_samples = 0
    _LOGGER.info('Counting samples in dialogs file.')
    with file_path.open() as file:
        for line in file:
            body = json.loads(line)
            if body.get('persona_0') and body.get('persona_1'):
                n_samples += 2
            else:
                n_samples += 1

    _LOGGER.info(f'There are dialog {n_samples} samples in {file_path}.')

    return n_samples
