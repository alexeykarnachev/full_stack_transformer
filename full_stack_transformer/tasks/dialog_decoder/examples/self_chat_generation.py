import pathlib

import torch

from full_stack_transformer.core.tokenizer import load_tokenizer_from_checkpoint
from full_stack_transformer.tasks.common.language_generator.generator import \
    LanguageGeneratorParams, LanguageGenerator
from full_stack_transformer.tasks.common.models.hf_gpt2 import load_model_from_checkpoint
from full_stack_transformer.tasks.dialog_decoder.text_input import DialogInput

if __name__ == '__main__':
    device = 'cuda:0'
    experiment_dir = pathlib.Path('../../../../data/experiments/pikabu_toloka_flibusta_v0/')
    ckpt_path = experiment_dir / 'models' / 'epoch=0_v1.ckpt'

    generator_params = LanguageGeneratorParams(
        max_number_of_generated_tokens=64,
        num_return_sequences=1,
        repetition_penalty=5.0,
        temperature=0.5,
        top_k=50,
        top_p=1.0
    )

    ckpt = torch.load(f=str(ckpt_path), map_location='cpu')
    model = load_model_from_checkpoint(ckpt=ckpt, device=device)
    tokenizer = load_tokenizer_from_checkpoint(ckpt=ckpt)
    generator = LanguageGenerator(
        model=model, eos_token_id=tokenizer.eos_token_id
    )

    persona_0 = "Я мужчина. Работаю ночным грузчиком."
    persona_1 = "Я женщина. Работаю космическим десантником."
    tags = "моё, абсурдная ситуация, день победы, врачи"
    persons = (persona_0, persona_1)
    utterances = ['Привет, что делал вчера?']

    generate_n_utterances = 20
    while generate_n_utterances != 0:
        print(f'Utterances left: {generate_n_utterances}')

        persona_idx = len(utterances) % 2
        persona = persons[persona_idx]

        dialog = DialogInput(
            utterances=utterances,
            persona=persona,
            persona_idx=persona_idx,
            tags=tags
        )

        inp_encoding = tokenizer.encode_for_inference(dialog)
        encodings = generator(encoding=inp_encoding, params=generator_params)
        text_samples = [tokenizer.decode_encoding(e) for e in encodings]
        utterances.append(text_samples[0])
        generate_n_utterances -= 1

    for speaker_idx, utterance in enumerate(utterances):
        print(f"Speaker {speaker_idx % 2}: {utterance}\n")
