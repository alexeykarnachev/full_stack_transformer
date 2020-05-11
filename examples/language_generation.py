import pathlib

import torch

from full_stack_transformer import (
    LanguageGenerator,
    LanguageGeneratorParams,
    Document,
    load_language_model_from_checkpoint,
    load_tokenizer_from_checkpoint
)

if __name__ == '__main__':
    device = 'cuda:0'
    experiment_dir = pathlib.Path('../data/experiments/nietzsche_v0/')
    ckpt_path = experiment_dir / 'models' / 'epoch=14.ckpt'

    generator_params = LanguageGeneratorParams(
        max_number_of_generated_tokens=64,
        num_return_sequences=8,
        repetition_penalty=3.0,
        temperature=0.5,
        top_k=50,
        top_p=1.0
    )

    ckpt = torch.load(f=str(ckpt_path), map_location='cpu')

    model = load_language_model_from_checkpoint(
        ckpt=ckpt, device=device, unlikelihood_alpha=None)

    tokenizer = load_tokenizer_from_checkpoint(
        ckpt=ckpt, max_body_len=64, max_meta_len=0)

    generator = LanguageGenerator(
        model=model,
        eos_token_id=tokenizer.eos_token_id
    )

    document = Document(body='Machine learning and neural networks are')

    inp_encoding = tokenizer.encode_document(
        document=document, with_eos=False
    )[0]

    out_encodings = generator(inp_encoding, params=generator_params)

    for enc in out_encodings:
        text = tokenizer.decode_encoding(enc)
        print(text + '\n\n')
