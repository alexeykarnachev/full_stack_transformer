"""Utilities for dynamic objects and classes imports."""

import pydoc


class FactoryError(Exception):
    pass


def get_object(class_path: str, *args, **kwargs):
    """Imports class and instantiate an object.

    Args:
        class_path (str):
            Class path of the required object to instantiate.

        *args:
            Class constructor arguments.

        **kwargs:
            Class constructor key value arguments.

    Returns:
        Object of a given class.
    """
    cls = get_class(class_path)

    obj = cls(*args, **kwargs)
    return obj


def get_class(class_path):
    """Imports and returns a class.

    Args:
        class_path (str):
            Class path of the required class object.

    Returns:
        Desired class object.
    """
    cls = pydoc.locate(class_path)
    if cls is None:
        raise FactoryError("Can't locate class: {}".format(class_path))
    return cls
