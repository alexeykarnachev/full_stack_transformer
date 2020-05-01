from typing import Optional, Sequence, List

import torch
import torch.nn.functional as F
import transformers
from pydantic import BaseModel

from lm_trainer.text_generator.generation_progress import GenerationProgress
from lm_trainer.text_generator.logits_modifiers import (
    IgnoredTokensModifier,
    RepetitiveTokensModifier,
    TemperatureModifier,
    TopKNucleusModifier
)
from lm_trainer.text_generator.model_handler import ModelHandler
from lm_trainer.tokenizers import Tokenizer


class TextGeneratorParams(BaseModel):
    seed_token_ids: Optional[Sequence[int]]
    ignored_token_ids: Optional[Sequence[int]]
    generation_max_len: int
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    num_return_sequences: int


class TextGenerator:
    def __init__(
            self,
            model: transformers.GPT2LMHeadModel,
            eos_token_id: int,
            tokenizer: Tokenizer
    ):
        self._model_handler = ModelHandler(model)
        self._eos_token_id = eos_token_id
        self._tokenizer = tokenizer

    def __call__(self, params: TextGeneratorParams) -> List[str]:
        generated_sequences = self._generate_sequences(params=params)
        generated_texts = self._decode_sequences(generated_sequences)
        return generated_texts

    def _generate_sequences(self, params: TextGeneratorParams):
        self._model_handler.eval()

        if params.seed_token_ids is None:
            seed_token_ids = [self._tokenizer.get_bos_token_id()]
        else:
            seed_token_ids = params.seed_token_ids

        input_ids = _prepare_model_input(
            sequence=seed_token_ids,
            device=self._model_handler.device,
            num_return_sequences=params.num_return_sequences)

        progress = GenerationProgress(
            eos_token_id=self._eos_token_id,
            max_length=params.generation_max_len)

        past = None

        generated_token_ids = torch.zeros(
            (input_ids.size()[0], params.generation_max_len), dtype=torch.long)
        generated_token_ids = generated_token_ids.to(self._model_handler.device)

        context_token_ids = input_ids.detach().clone()

        while not progress.finished:
            logits, past = self._model_handler(input_ids, past)

            next_token_logits = logits[:, -1, :]
            token_ids_to_penalize = torch.cat(
                [context_token_ids, generated_token_ids], 1)
            _modify_next_token_logits(
                next_token_logits=next_token_logits,
                ignored_token_ids=params.ignored_token_ids,
                token_ids_to_penalize=token_ids_to_penalize,
                repetition_penalty=params.repetition_penalty,
                temperature=params.temperature,
                top_k=params.top_k,
                top_p=params.top_p)
            next_token_ids = _sample_next_token_ids(next_token_logits)
            progress.update(next_token_ids)

            generated_token_ids[:, progress.current_length - 1] = next_token_ids

            input_ids = next_token_ids.unsqueeze(1)

        generated_sequences = _get_generated_sequences(
            generated_tokens=generated_token_ids,
            generated_sample_lengths=progress.generated_sample_lengths)

        return generated_sequences

    def _decode_sequences(
            self, generated_sequences: List[List[int]]
    ) -> List[str]:
        decoded_sequences = self._tokenizer.decode_batch(generated_sequences)
        cleaned_texts = []

        for sequence in decoded_sequences:
            cleaned_sequence = self._tokenizer.clean_after_generation(
                string=sequence,
                remove_bos_eos=True)
            cleaned_texts.append(cleaned_sequence)

        return cleaned_texts


def _get_generated_sequences(generated_tokens, generated_sample_lengths):
    generated_sequences = []
    for i in range(generated_tokens.size()[0]):
        seq = generated_tokens[i, :generated_sample_lengths[i]]
        seq = seq.detach().cpu().numpy().tolist()
        generated_sequences.append(seq)

    return generated_sequences


def _prepare_model_input(sequence, device, num_return_sequences):
    tensor = torch.tensor(sequence).unsqueeze(0).to(device)

    batch_size = tensor.shape[0]
    cur_len = tensor.shape[1]

    tensor = tensor.unsqueeze(1).expand(
        batch_size, num_return_sequences, cur_len)
    tensor = tensor.contiguous().view(
        batch_size * num_return_sequences, cur_len)

    return tensor


def _modify_next_token_logits(
        next_token_logits,
        ignored_token_ids,
        token_ids_to_penalize,
        repetition_penalty,
        temperature,
        top_k,
        top_p
):
    modifiers = [
        IgnoredTokensModifier(
            ignored_token_ids=ignored_token_ids
        ),
        RepetitiveTokensModifier(
            penalty=repetition_penalty,
            token_ids_to_penalize=token_ids_to_penalize
        ),
        TemperatureModifier(
            temperature=temperature
        ),
        TopKNucleusModifier(
            top_k=top_k,
            top_p=top_p)
    ]

    _ = [modifier(next_token_logits) for modifier in modifiers]


def _sample_next_token_ids(next_token_logits: torch.tensor) -> torch.tensor:
    next_tokens = torch.multinomial(
        F.softmax(next_token_logits, dim=-1), num_samples=1).squeeze(1)
    return next_tokens
