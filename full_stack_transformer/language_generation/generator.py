from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional

from full_stack_transformer.language_generation.logits_modifiers import (
    IgnoredTokensModifier,
    RepetitiveTokensModifier,
    TemperatureModifier,
    TopKNucleusModifier
)
from full_stack_transformer.language_generation.progress_tracker import \
    GenerationProgressTracker
from full_stack_transformer.language_modelling.data_structures import \
    DocumentEncoding, LanguageModelInput
from full_stack_transformer.language_modelling.encodings_collate import \
    DocumentEncodingsCollate
from full_stack_transformer.language_modelling.modelling.model import \
    LanguageModel


@dataclass
class LanguageGeneratorParams:
    max_number_of_generated_tokens: int
    num_return_sequences: int

    repetition_penalty: float
    temperature: float
    top_k: float
    top_p: float


@dataclass
class ReplyCandidate:
    token_ids: Sequence[int]


class LanguageGenerator:
    def __init__(self, model: LanguageModel, eos_token_id: int):
        self._model = model
        self._eos_token_ids = eos_token_id

        # LanguageGenerator never needs to pad sequences, so the `pad_value`
        # could be any here (e.g. 0).
        self._collator = DocumentEncodingsCollate(pad_value=0)

    def __call__(
            self,
            encoding: DocumentEncoding,
            params: LanguageGeneratorParams
    ) -> Sequence[ReplyCandidate]:
        encodings = [encoding] * params.num_return_sequences
        model_inp = self._collator(encodings=encodings)

        progress = GenerationProgressTracker(
            eos_token_id=self._eos_token_ids,
            max_length=params.max_number_of_generated_tokens
        )

        generated_token_ids = torch.zeros(
            params.num_return_sequences,
            params.max_number_of_generated_tokens,
            dtype=torch.long
        )

        generated_token_ids = generated_token_ids.to(self._model.device)

        past_token_ids = model_inp.input_ids.detach().clone()
        not_eos_mask = ~(past_token_ids == self._eos_token_ids).all(0)
        past_token_ids = past_token_ids[:, not_eos_mask]

        while not progress.finished:
            model_out = self._model.infer(inp=model_inp)
            next_token_logits = model_out.logits[:, -1, :]
            past_token_ids = torch.cat(
                tensors=[past_token_ids, generated_token_ids],
                dim=1
            )
            _modify_next_token_logits(
                next_token_logits=next_token_logits,
                ignored_token_ids=[],
                token_ids_to_penalize=past_token_ids,
                repetition_penalty=params.repetition_penalty,
                temperature=params.temperature,
                top_k=params.top_k,
                top_p=params.top_p
            )
            next_token_ids = _sample_next_token_ids(next_token_logits)
            progress.update(next_token_ids)

            generated_token_ids[:, progress.current_length - 1] = next_token_ids

            input_ids = next_token_ids.unsqueeze(1)
            token_type_ids = model_inp.token_type_ids[:, -1:]

            model_inp = LanguageModelInput(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                past=model_out.past
            )

        candidates = _get_candidates(
            generated_tokens=generated_token_ids,
            generated_sample_lengths=progress.generated_sample_lengths
        )

        return candidates


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
            top_p=top_p
        )
    ]

    _ = [modifier(next_token_logits) for modifier in modifiers]


def _sample_next_token_ids(next_token_logits: torch.tensor) -> torch.tensor:
    probabilities = torch.nn.functional.softmax(
        input=next_token_logits,
        dim=-1
    )
    next_tokens = torch.multinomial(
        probabilities, num_samples=1
    )
    return next_tokens.squeeze(1)


def _get_candidates(generated_tokens, generated_sample_lengths):
    candidates = []
    for i in range(generated_tokens.size()[0]):
        token_ids = generated_tokens[i, :generated_sample_lengths[i]]
        token_ids = token_ids.detach().cpu().numpy().tolist()
        candidate = ReplyCandidate(token_ids=token_ids)
        candidates.append(candidate)

    return candidates
