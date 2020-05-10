import inspect
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import List, Any

from docstring_parser.google import parse


@dataclass
class DocstringArgument:
    name: str
    annotation: Any
    description: str
    required: bool
    default: Any


def parse_docstring_arguments(obj) -> List[DocstringArgument]:
    parsed_doc = parse(obj.__doc__)
    signature = inspect.signature(obj)
    parsed_arguments = []
    for doc_param in parsed_doc.params:
        name = doc_param.arg_name
        param_sign = signature.parameters[name]
        annotation = param_sign.annotation
        description = doc_param.description
        required = not doc_param.is_optional
        default = param_sign.default

        argument = [name, annotation, description, required, default]
        argument = DocstringArgument(*argument)

        parsed_arguments.append(argument)

    return parsed_arguments


class ArgparserExtender:
    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser) -> ArgumentParser:
        arguments = parse_docstring_arguments(cls.__init__)

        for arg in arguments:
            parser.add_argument(
                f'--{arg.name}',
                required=arg.required,
                type=arg.annotation,
                help=arg.description,
                default=arg.default
            )

        return parser
