import random

import string

LETTERS = string.ascii_letters + '\n'


def generate_random_text_file(file_path, size):
    chars = ''.join([random.choice(LETTERS) for _ in range(size)])

    with open(file_path, 'w') as f:
        f.write(chars)
