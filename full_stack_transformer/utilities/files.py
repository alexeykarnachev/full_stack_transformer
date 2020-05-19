import hashlib
import pathlib


def count_lines_in_file(file_path: pathlib.Path) -> int:
    number_of_lines = 0
    with file_path.open() as file:
        for line in file:
            if line != '\n':
                number_of_lines += 1

    return number_of_lines


def get_file_md5(file_path: pathlib.Path) -> str:
    """Calculates file data md5 hash.

    Args:
        file_path: Path to the file.

    Returns:
        File content hash string.
    """
    hash_md5 = hashlib.md5()
    with file_path.open('rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
