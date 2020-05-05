import hashlib


def get_string_md5(string: str) -> str:
    """Gets input string md5 hexdigest hash."""
    return hashlib.md5(string.encode()).hexdigest()