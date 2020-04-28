from typing import List, Optional

import torch
import torch.nn.functional as F
import transformers

from chitchat.text_generator.generation_progress import GenerationProgress
from chitchat.text_generator.logits_modifiers import (
    RepetitiveTokensModifier,
    TemperatureModifier,
    TopKNucleusModifier, IgnoredTokensModifier,
)
from chitchat.text_generator.model_handler import ModelHandler


class TextGenerator:
    def __init__(self,
                 model: transformers.GPT2DoubleHeadsModel,
                 eos_token_id: int):
        self._model_handler = ModelHandler(model)
        self._eos_token_id = eos_token_id

    def __call__(
            self,
            sequence: List[int],
            token_types: List[int],
            ignored_token_ids: Optional[List[int]],
            generation_max_len: int,
            temperature: float,
            top_k: int,
            top_p: float,
            repetition_penalty: float,
            num_return_sequences: int,
            penalize_candidates: bool
    ):
        self._model_handler.eval()

        input_ids, token_type_ids = _prepare_model_input(
            sequence=sequence,
            token_types=token_types,
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
            with torch.no_grad():
                output = self._model_handler(input_ids, token_type_ids, past)

            next_token_logits = output[0][:, -1, :]
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

            token_type_ids = token_type_ids[:, -1:]
            input_ids = next_token_ids.unsqueeze(1)
            past = output[2]

        generated_sequences = _get_generated_sequences(
            generated_tokens=generated_token_ids,
            generated_sample_lengths=progress.generated_sample_lengths
        )

        mc_scores = _calc_mc_scores(
            model_handler=self._model_handler,
            last_token_ids=[seq[-1] for seq in generated_sequences],
            last_token_type_ids=token_type_ids[:, -1],
            past=past
        )

        scored_candidates = [
            (seq.detach().cpu().numpy(), score)
            for seq, score in zip(generated_sequences, mc_scores)
        ]
        scored_candidates = sorted(scored_candidates, key=lambda x: -x[1])

        return scored_candidates


def _get_generated_sequences(generated_tokens, generated_sample_lengths):
    generated_sequences = []
    for i in range(generated_tokens.size()[0]):
        seq = generated_tokens[i, :generated_sample_lengths[i]]
        generated_sequences.append(seq)

    return generated_sequences


def _calc_mc_scores(model_handler, last_token_ids, last_token_type_ids, past):
    mc_scores = []
    for i in range(len(last_token_ids)):
        _input_ids = last_token_ids[i].unsqueeze(0).unsqueeze(0)
        _token_type_ids = last_token_type_ids[i].unsqueeze(0).unsqueeze(0)
        _past = [p[:, i, ...].unsqueeze(1) for p in past]
        _, mc_prediction_scores, _, _ = model_handler(
            _input_ids, _token_type_ids, _past)
        mc_scores.append(mc_prediction_scores[0][0].item())

    return mc_scores


def _prepare_model_input(sequence, token_types, device, num_return_sequences):
    model_input = []
    for input_seq in [sequence, token_types]:
        tensor = torch.tensor(input_seq).unsqueeze(0).to(device)

        batch_size = tensor.shape[0]
        cur_len = tensor.shape[1]

        tensor = tensor.unsqueeze(1).expand(
            batch_size, num_return_sequences, cur_len)
        tensor = tensor.contiguous().view(
            batch_size * num_return_sequences, cur_len)

        model_input.append(tensor)
    return model_input


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
