import glob
import hashlib
import json
import os
import pathlib
import pickle
from typing import Any, Mapping

from full_stack_transformer.utilities.json_encoder import CustomJsonEncoder


class FileIOError(Exception):
    pass


def dump_json(obj: Any, file_path: pathlib.Path):
    """Serializes an object to json file."""
    with file_path.open('w') as file:
        json.dump(
            obj=obj,
            fp=file,
            cls=CustomJsonEncoder,
            indent=2,
            ensure_ascii=False)


def load_json(file_path: pathlib.Path):
    """Reads json file into a python object."""
    with file_path.open('r') as f:
        return json.load(f)


def get_file_md5(file_path: pathlib.Path) -> str:
    """Calculates file data md5 hash."""
    hash_md5 = hashlib.md5()
    with file_path.open('rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_object(file_path: pathlib.Path) -> Any:
    """Loads pickled object from file."""
    with file_path.open('rb') as handle:
        return pickle.load(handle)


def dump_object(obj: Any, file_path: pathlib.Path):
    """Dumps python object into a pickle file."""
    with file_path.open('wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def prepare_dataset_dir(
        datasets_root: pathlib.Path,
        dataset_name: str,
        description: Mapping,
        version_prefix: str = '',
        max_version: int = 1000
) -> pathlib.Path:
    """Prepares dataset directory.

    This function creates a directory:
    `datasets_root/dataset_name/dataset_version`, where the `dataset_version` is
    constructed as follows: `f'{version_prefix}_{version_id}'`.

    Args:
        datasets_root:
            Root dir, where dataset folder will be saved.

        dataset_name:
            Dataset folder name.

        description:
            Dict which will be dumped as a description json.

        version_prefix:
            Prefix which will be appended to the dataset dir name.

        max_version:
            The maximum version number. It determines the max number of the
            dataset subdirectories, placed under the one
            `datasets_root/dataset_name` dir.

    Returns: Dataset directory path.
    """
    dataset_dir: pathlib.Path = datasets_root / dataset_name
    dataset_dir.mkdir(exist_ok=True, parents=True)

    version_id = 0
    while (dataset_dir / f'{version_prefix}{version_id}').is_dir():
        version_id += 1
        if version_id > max_version:
            raise FileIOError(
                f'Too many dataset versions. Clean the {dataset_dir} directory.'
            )

    dataset_dir = dataset_dir / f'{version_prefix}{version_id}'
    dataset_dir.mkdir(exist_ok=True, parents=False)

    dump_json(description, dataset_dir / 'description.json')
    return dataset_dir


def get_file_by_time(
        directory: pathlib.Path, extension: str, how='max'
) -> pathlib.Path:
    """Obtains filepath from the specific directory by the creation time
    condition.

    Args:
        directory:
            Path to the directory to search files in.
        extension:
            Search file only with this extension.
        how:
            The condition to select file by creation datetime (max or min).

    Returns:
        Full path to the file.
    """
    funcs = {'max': max, 'min': min}
    list_of_files = glob.glob(str(directory / f'*.{extension}'))
    latest_file = funcs[how](list_of_files, key=os.path.getctime)
    return pathlib.Path(latest_file)
