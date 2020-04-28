from typing import List, Optional

import torch
import torch.nn.functional as F
import transformers

from lm_trainer.text_generator.generation_progress import GenerationProgress
from lm_trainer.text_generator.logits_modifiers import (
    IgnoredTokensModifier,
    RepetitiveTokensModifier,
    TemperatureModifier,
    TopKNucleusModifier
)
from lm_trainer.text_generator.model_handler import ModelHandler


class TextGenerator:
    def __init__(self,
                 model: transformers.GPT2LMHeadModel,
                 eos_token_id: int):
        self._model_handler = ModelHandler(model)
        self._eos_token_id = eos_token_id

    def __call__(
            self,
            sequence: List[int],
            ignored_token_ids: Optional[List[int]],
            generation_max_len: int,
            temperature: float,
            top_k: int,
            top_p: float,
            repetition_penalty: float,
            num_return_sequences: int
    ):
        self._model_handler.eval()

        input_ids = _prepare_model_input(
            sequence=sequence,
            device=self._model_handler.device,
            num_return_sequences=num_return_sequences)

        progress = GenerationProgress(
            eos_token_id=self._eos_token_id,
            max_length=generation_max_len)

        past = None

        generated_token_ids = torch.zeros(
            (input_ids.size()[0], generation_max_len)
        ).long().to(self._model_handler.device)

        context_token_ids = input_ids.detach().clone()

        while not progress.finished:
            logits, past = self._model_handler(input_ids, past)

            next_token_logits = logits[:, -1, :]
            token_ids_to_penalize = torch.cat(
                [context_token_ids, generated_token_ids], 1)
            _modify_next_token_logits(
                next_token_logits=next_token_logits,
                ignored_token_ids=ignored_token_ids,
                token_ids_to_penalize=token_ids_to_penalize,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            next_token_ids = _sample_next_token_ids(next_token_logits)
            progress.update(next_token_ids)

            generated_token_ids[:, progress.current_length - 1] = next_token_ids

            input_ids = next_token_ids.unsqueeze(1)

        generated_sequences = _get_generated_sequences(
            generated_tokens=generated_token_ids,
            generated_sample_lengths=progress.generated_sample_lengths
        )

        return generated_sequences


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
    IgnoredTokensModifier(
        ignored_token_ids=ignored_token_ids
    )(next_token_logits)

    RepetitiveTokensModifier(
        penalty=repetition_penalty,
        token_ids_to_penalize=token_ids_to_penalize
    )(next_token_logits)

    TemperatureModifier(
        temperature=temperature
    )(next_token_logits)

    TopKNucleusModifier(
        top_k=top_k,
        top_p=top_p
    )(next_token_logits)


def _sample_next_token_ids(next_token_logits: torch.tensor) -> torch.tensor:
    next_tokens = torch.multinomial(
        F.softmax(next_token_logits, dim=-1), num_samples=1).squeeze(1)
    return next_tokens
