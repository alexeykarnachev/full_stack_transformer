import pathlib

import torch

from full_stack_transformer.tasks.document_lm import (
    LanguageGeneratorParams,
    load_model_from_checkpoint,
    load_tokenizer_from_checkpoint,
    LanguageGenerator,
    DocumentInput
)

if __name__ == '__main__':
    device = 'cuda:0'
    experiment_dir = pathlib.Path('../../../../data/experiments/nietzsche_v0/')
    ckpt_path = experiment_dir / 'models' / 'epoch=6.ckpt'

    generator_params = LanguageGeneratorParams(
        max_number_of_generated_tokens=64,
        num_return_sequences=8,
        repetition_penalty=3.0,
        temperature=1.0,
        top_k=50,
        top_p=1.0
    )

    ckpt = torch.load(f=str(ckpt_path), map_location='cpu')
    model = load_model_from_checkpoint(ckpt=ckpt, device=device)
    tokenizer = load_tokenizer_from_checkpoint(ckpt=ckpt)
    generator = LanguageGenerator(
        model=model, eos_token_id=tokenizer.eos_token_id
    )

    document = DocumentInput(body='The best filosopher of the 19th century is')
    inp_encoding = tokenizer.encode_for_inference(text_input=document)[0]

    out_encodings = generator(inp_encoding, params=generator_params)
    for enc in out_encodings:
        text = tokenizer.decode_encoding(enc)
        print(text + '\n\n')
