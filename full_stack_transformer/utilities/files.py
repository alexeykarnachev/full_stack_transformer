import pathlib


def count_lines_in_file(file_path: pathlib.Path) -> int:
    number_of_lines = 0
    with file_path.open() as file:
        for line in file:
            if line != '\n':
                number_of_lines += 1

    return number_of_lines
