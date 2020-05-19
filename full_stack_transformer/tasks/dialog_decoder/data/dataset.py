import json
import logging
import pathlib
import tempfile

from full_stack_transformer.core.data.dataset import Dataset
from full_stack_transformer.core.data.encodings_collate import EncodingsCollate
from full_stack_transformer.tasks.dialog_decoder.data.text_lines_parser import DialogLinesParser
from full_stack_transformer.tasks.dialog_decoder.tokenizer import DialogTokenizer
from full_stack_transformer.utilities.files import get_file_md5

_LOGGER = logging.getLogger(__name__)


class DialogDataset(Dataset):
    def __init__(
            self,
            file_path: pathlib.Path,
            tokenizer: DialogTokenizer,
    ):
        collate = EncodingsCollate(pad_value=tokenizer.pad_token_id)
        text_lines_parser = DialogLinesParser()

        super().__init__(
            file_path=file_path,
            tokenizer=tokenizer,
            text_lines_parser=text_lines_parser,
            encodings_collate=collate,
            chunk_size=2000
        )

    def __len__(self) -> int:
        return _count_samples_in_file(self._file_path)


def _count_samples_in_file(file_path: pathlib.Path) -> int:
    file_hash = get_file_md5(file_path=file_path)
    tmp_dir = pathlib.Path(tempfile.gettempdir())
    out_file = tmp_dir / file_hash

    if out_file.is_file():
        with out_file.open() as file:
            content = file.read().strip()
            n_samples = int(content)
    else:
        n_samples = 0
        _LOGGER.info('Counting samples in dialogs file.')
        with file_path.open() as file:
            for line in file:
                body = json.loads(line)
                if body.get('persona_0') and body.get('persona_1'):
                    n_samples += 2
                else:
                    n_samples += 1

        _LOGGER.info(f'There are {n_samples} dialog samples in {file_path}')

        with out_file.open('w') as file:
            file.write(str(n_samples))

    return n_samples
