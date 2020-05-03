![Logo](docs/source/_images/logos/lightning_logo.svg)

# Full Stack Transformer
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
For this example, files are already placed here:
```
data/documents/nietzsche/
├── train.txt
└── valid.txt
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

We have the dataset, now we are ready to train the model.

### Train Model
The library uses `pytorch-lightning` for training and arguments which are used by
lightning `Trainer` class are allowed as a command line argument for the script below.

To check them all execute:
```
python full_stack_transformer/scripts/train_model.py --help
```

Now, let's train the model:
```
python full_stack_transformer/scripts/train_model.py \
--seed 228 \
--dataset_dir data/datasets/nietzsche/0 \
--model_path gpt2 \
--batch_size 4 \
--learning_rate 2.5e-05 \
--num_warmup_steps 200 \
--num_cycles 1 \
--gradient_clip_val 5.0 \
--accumulate_grad_batches 4 \
--max_epochs 10 \
--val_check_interval 330 \
--gpus "0," \
--log_text_samples
```

If you don't have gpt2 model downloaded, it'll be obtained from the huggingface server (548M).

The training has been started. All experiment related files a placed in the experiment directory:
```
data/experiments/nietzsche/0/
├── _ckpt_epoch_0.ckpt
├── _ckpt_epoch_1.ckpt
├── description.json
├── generated.txt
├── logs
│   ├── critical.log
│   ├── debug.log
│   ├── errors.log
│   └── info.log
└── text_generator_params.json
```

### Monitor training
Run `tensorboard`:
```
tensorboard --logdir=./data/tb_logs/ --port=6006
```
TensorBoard interface is available here: [http://localhost:6006/](http://localhost:6006/)
<br>
![Logo](docs/source/_images/tb_example.png)


Also, text samples are generated during the training on each validation step.
They are logged here:
```
cat data/experiments/nietzsche/0/generated.txt
```
```
...
{
    "Global step": 250,
    "Current epoch": 2,
    "Generator params": {
        "seed_text": null,
        "ignored_words": null,
        "generation_max_len": 64,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 1.0,
        "repetition_penalty": 5.0,
        "num_return_sequences": 16
    },
    "Error message": null,
    "Generated samples": [
        "and\nthe greatest of all philosophers.",
...
```

Text generator parameters could be changed here:
```text
cat data/experiments/nietzsche/0/text_generator_params.json
```

Changed parameters will be applied on the next validation step.

### Serve application
### Serve telegram bot

##Inference

##How to Implement new Tokenizer


