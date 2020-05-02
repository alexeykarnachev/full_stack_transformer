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

## End-to-end Steps
These are steps to prepare dataset, train a model and serve it.
### Prepare dataset

First, you need two text (train and validation) files which contain raw documents
([example](data/documents/ru_rap/train.txt)). It's a simple text file with documents, which are
separated by `|` symbol.

Now, run dataset preparation script:
<br>
```bash
python lm_trainer/scripts/prepare_dataset.py \
--documents_train_file path/to/your/raw/documents/train/file.txt \
--documents_valid_file path/to/your/raw/documents/valid/file.txt \
--end_of_document end_of_document_separation_string \
--tokenizer_cls_name RuTransformersTokenizer \
--max_sample_length 256 \
--min_sample_length 8 \
--datasets_root path/to/dir/where/all/your/datasets/are/ \
--dataset_name name_of_the_dataset
```

Arguments:
- **--documents_train_file** and **--documents_valid_file** are just paths to your 
train and validation raw text files
- **--end_of_document** special string in your raw files which indicates the end
of a document. Must be placed on a new line, so no other symbols in this line are allowed
(except `\n` on the end of it)
- **--tokenizer_cls_name**. Each model requires its specific tokenization logic.
For instance, if you want to use pre-trained weights of the 
[ru_transformers](https://github.com/mgrankin/ru_transformers) model, you definitely need
to use their tokenizer. `lm_trainer` wraps this tokenizer in the `RuTransformersTokenizer` class.
So, in such a case, you can just pass a name of this class to the argument.
If you want to use another model with another tokenizer, make sure, that this tokenizer
could is importable from `lm_trainer.tokenization` package. But for now, only the
`RuTransformersTokenizer` is available. If you want to implement wrapper for another
tokenizer, check the [How to Implement new Tokenizer](#how-to-implement-new-tokenizer)
- **--max_sample_length** represents the maximum sequence length (number of tokens). Make
sure, that it is not greater than your transformer model can handle. If there are a documents
in your dataset, that are longer than this number, they will be split on appropriate lengths
sequences
- **--min_sample_length** indicates the minimum sample length. Documents that are shorter than
this number won't be added to the dataset
- **--datasets_root** is a directory path, where all your datasets are stored. The new dataset
sub-directory will be created there
- **--dataset_name** is the name of your dataset. Just place any name you want (or skip it).

### Train Model
### Monitor training
### Serve application
### Serve telegram bot

##Inference

##How to Implement new Tokenizer


