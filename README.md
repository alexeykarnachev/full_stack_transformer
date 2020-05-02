![Logo](docs/source/_images/logos/lightning_logo.svg)

# Language Models Trainer
Pytorch library for end-to-end transformer models training, inference and serving.
<br>
<br>
Powered by:
- [tokenizers](https://github.com/huggingface/tokenizers) fast tokenization and dataset preparation
- [transformers](https://github.com/huggingface/transformers) model backbones
- [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) training process
- [fastapi](https://github.com/tiangolo/fastapi) application serving
- [aiogram](https://github.com/aiogram/aiogram) telegram bot serving

## End-to-end Example
As they say, it’s better to see once...<br>
So, let's go through the whole pipeline from dataset building to the application
serving on the [nietzsche](data/documents/nietzsche) texts example.

### Prepare dataset

First, you need two text files (train and validation) which contain raw documents.

```
ls data/documents/nietzsche
```
`train.txt` and `valid.txt` are there.

Documents in these files are separated by a delimiter. In this example it's the
`[END_OF_DOCUMENT]` string.
```
tail -n5 data/documents/nietzsche/train.txt
```

Here is the content:
```
force of the latter forces triumph for the former.
[END_OF_DOCUMENT]
123
[END_OF_DOCUMENT]
=The Breaking off of Churches.=--There is not sufficient religion in the
world merely to put an end to the number of religions.
[END_OF_DOCUMENT]
```

Files are in place and we are ready to prepare the dataset:
```
python full_stack_transformer/scripts/prepare_dataset.py \
--documents_train_file data/documents/nietzsche/train.txt \
--documents_valid_file data/documents/nietzsche/valid.txt \
--end_of_document "[END_OF_DOCUMENT]" \
--tokenizer_cls_name GPT2Tokenizer \
--max_sample_length 128 \
--min_sample_length 8 \
--datasets_root data/datasets \
--dataset_name nietzsche
```

The dataset has been created, here is its directory structure:
```
data/datasets/nietzsche/
└── 0
    ├── description.json
    ├── train
    │   ├── corpus.npy
    │   └── sample_start_positions.npy
    └── valid
        ├── corpus.npy
        └── sample_start_positions.npy
```

Let's check the description:
```
cat data/datasets/nietzsche/0/description.json
```
```json
{
  "documents_train_file": "data/documents/nietzsche/train.txt",
  "documents_valid_file": "data/documents/nietzsche/valid.txt",
  "end_of_document": "[END_OF_DOCUMENT]",
  "tokenizer_cls_name": "GPT2Tokenizer",
  "max_sample_length": 128,
  "min_sample_length": 8,
  "datasets_root": "data/datasets",
  "dataset_name": "nietzsche",
  "number_of_train_samples": 1347,
  "number_of_train_tokens": 135094,
  "number_of_valid_samples": 69,
  "number_of_valid_tokens": 7636
}
```
Now we are ready to train the model.


### Train Model
### Monitor training
### Serve application
### Serve telegram bot

##Inference

##How to Implement new Tokenizer


