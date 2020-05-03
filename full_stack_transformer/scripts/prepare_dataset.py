import argparse
import pathlib
from collections import defaultdict

from full_stack_transformer.datasets.documents_dataset import DocumentsDatasetReader
from full_stack_transformer.tokenization import get_tokenizer
from full_stack_transformer.utilities.file_io import prepare_dataset_dir

THIS_DIR = pathlib.Path(__file__).parent


def _parse_args():
    ds_root = THIS_DIR / '../../data/datasets'

    parser = argparse.ArgumentParser(
        description='Script which prepares train and validation dataset files.')

    parser.add_argument(
        '--documents_train_file', type=_str2path, required=True,
        help='Input raw documents train file. Each document must end with '
             'a line which contains only `end_of_document` separator.'
    )
    parser.add_argument(
        '--documents_valid_file', type=_str2path, required=True,
        help='Input raw documents validation file. Each document must end with '
             'a line which contains only `end_of_document` separator.'
    )
    parser.add_argument(
        '--end_of_document', type=str, required=True,
        help='String line which must be presented at the end of each document '
             'in the documents file.'
    )
    parser.add_argument(
        '--tokenizer_cls_name', type=str, required=True,
        choices=['RuTransformersTokenizer', 'GPT2Tokenizer'],
        help='Class name of the tokenizer object. This tokenizer name must be '
             'importable from `full_stack_transformer.tokenization`.'
    )
    parser.add_argument(
        '--max_sample_length', type=int, required=True,
        help="The maximum length of a dataset sample. If some document is "
             "larger than this number, it'll be split on smaller samples."
    )
    parser.add_argument(
        '--min_sample_length', type=int, required=False, default=10,
        help="The minimum length of a dataset sample. If some sample is smaller"
             "than this number, it'll be dropped. Default is: 10."
    )
    parser.add_argument(
        '--datasets_root', type=_str2path, required=False, default=ds_root,
        help=f'Root dir with datasets sub-dirs. Default is: {ds_root}'
    )
    parser.add_argument(
        '--dataset_name', type=str, required=False, default='default',
        help=f'Dataset name. Default is: "default".'
    )

    args = parser.parse_args()
    return args


def _str2path(path: str) -> pathlib.Path:
    return pathlib.Path(path)


def main():
    args = _parse_args()
    tokenizer = get_tokenizer(args.tokenizer_cls_name)
    description = defaultdict()
    description.update(args.__dict__)

    datasets = dict()
    dataset_names = ['train', 'valid']
    for name in dataset_names:
        file_path = getattr(args, f'documents_{name}_file')
        dataset_reader = DocumentsDatasetReader(
            file_path=file_path,
            end_of_document=args.end_of_document,
            tokenizer=tokenizer,
            max_sample_length=args.max_sample_length,
            min_sample_length=args.min_sample_length)

        dataset = dataset_reader.construct()
        datasets[name] = dataset

        description[f'number_of_{name}_samples'] = dataset.number_of_samples
        description[f'number_of_{name}_tokens'] = dataset.number_of_tokens

    dataset_dir = prepare_dataset_dir(
        datasets_root=args.datasets_root,
        dataset_name=args.dataset_name,
        description=description)

    for name in dataset_names:
        dataset_sub_dir: pathlib.Path = dataset_dir / name
        dataset_sub_dir.mkdir(exist_ok=False, parents=False)
        datasets[name].save(dir_path=dataset_sub_dir)


if __name__ == '__main__':
    main()
