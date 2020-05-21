import json
import pathlib
from typing import Mapping

from full_stack_transformer.utilities.arguments import ArgparserExtender
from full_stack_transformer.utilities.json_encoder import JsonEncoder
from full_stack_transformer.utilities.log_config import prepare_logging


class Workspace(ArgparserExtender):
    """Experiment workspace, which handles files and folders creation."""

    _LOGS_DIR_NAME = 'logs'
    _MODELS_DIR_NAME = 'models'
    _MAX_VERSION = 63

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return self._version

    @property
    def logs_dir(self):
        return self._logs_dir

    @property
    def models_dir(self):
        return self._models_dir

    @property
    def experiment_dir(self):
        return self._main_dir

    def __init__(
            self,
            experiments_root: str,
            experiment_name: str = 'default'
    ):
        """
        Args:
            experiments_root (str):
                Root dir where all experiments folders are stored.

            experiment_name (str, optional):
                Base name of the experiment. Experiment folder name also will
                contain a version number postfix. Defaults to default.
        """

        self._root = pathlib.Path(experiments_root)
        self._name = experiment_name

        self._version = self._get_version()
        self._main_dir = self._get_main_dir(self._version)
        self._logs_dir = self._main_dir / self._LOGS_DIR_NAME
        self._models_dir = self._main_dir / self._MODELS_DIR_NAME

        self._main_dir.mkdir(exist_ok=True, parents=True)
        self._logs_dir.mkdir(exist_ok=True, parents=True)
        self._models_dir.mkdir(exist_ok=True, parents=True)

        prepare_logging(logs_dir=self._logs_dir)

    def _get_main_dir(self, version: int):
        return self._root / f'{self._name}_v{version}'

    def _get_version(self) -> int:
        version = 0

        while self._get_main_dir(version).is_dir():
            version += 1

            if version > self._MAX_VERSION:
                raise ValueError(
                    f'Too many experiments with name: {self._name}. Please, '
                    f'remove old ones from {self._root}'
                )

        return version

    def save_json(self, name: str, content: Mapping):
        with (self._main_dir / name).open('w') as file:
            json.dump(
                content, file, ensure_ascii=False, indent=2, cls=JsonEncoder)

    def save_description(self, content: Mapping):
        self.save_json(name='description.json', content=content)
