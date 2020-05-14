import abc
from typing import Sequence, Optional

import torch
from transformers import modeling_utils

_MINUS_INF = -float("Inf")


class NextTokenLogitsModifierError(Exception):
    pass


class NextTokenLogitsModifier(abc.ABC):
    @abc.abstractmethod
    def modify(self, logits: torch.tensor):
        """Modifies next token logits. Should modify them inplace."""

    def __call__(self, logits: torch.tensor):
        return self.modify(logits)


class TemperatureModifier(NextTokenLogitsModifier):
    """Classic temperature logits distribution modifier."""

    def __init__(self, temperature: float):
        """
        Args:
            temperature:
                The value used to module the next token probabilities.
                Must be >= 0. 0 Means, that the token with maximum logits value
                will be taken (no sampling).
        Raises:
            NextTokenLogitsModifierError: in case of incorrect input arguments.
        """
        self._temperature = temperature
        self._check_arguments_validity()

    def _check_arguments_validity(self) -> None:
        if self._temperature < 0:
            raise NextTokenLogitsModifierError('`temperature` must be >= 0.')

    def modify(self, logits: torch.tensor):
        logits.mul_(1 / self._temperature)


class TopKNucleusModifier(NextTokenLogitsModifier):
    """Filters a distribution of logits using top-k and/or top-p filtering"""

    def __init__(self, top_k: int, top_p: float):
        """
        Args:
            top_k:
                The number of highest probability vocabulary tokens to keep for
                top-k-filtering. Between 0 and infinity. If 0, top-k filtering
                will not be using.
            top_p:
                The cumulative probability of parameter highest probability
                vocabulary tokens to keep for nucleus sampling. Must be between
                0 and 1.
        Notes:
            `top_k` is performed before `top_p`. It means, that cumulative
            probability will be counted only for top k tokens.
        Raises:
            NextTokenLogitsModifierError: in case of incorrect input arguments.
        """
        self._top_k = top_k
        self._top_p = top_p

        self._check_arguments_validity()

    def _check_arguments_validity(self) -> None:
        if self._top_k < 0:
            raise NextTokenLogitsModifierError(
                '`top_k` must be >= 0. Use `top_k` = 0 if you want to switch '
                'off the top-k filtering.')
        elif not (0 <= self._top_p <= 1):
            raise NextTokenLogitsModifierError(
                '`top_p` must be between 0 and 1.')

    def modify(self, logits: torch.tensor):
        """Delegates call to the transformers `top_k_top_p_filtering` func."""
        modeling_utils.top_k_top_p_filtering(
            logits=logits,
            top_k=self._top_k,
            top_p=self._top_p,
            filter_value=_MINUS_INF)


class IgnoredTokensModifier(NextTokenLogitsModifier):
    """Assigns zero probabilities logits to the ignored tokens."""

    def __init__(self, ignored_token_ids: Optional[Sequence[int]]):
        """
        Args:
            ignored_token_ids:
                Ignored token indexes sequence.
        """
        self._ignored_token_ids = list(set(ignored_token_ids))

    def modify(self, logits: torch.tensor):
        if self._ignored_token_ids:
            logits[:, self._ignored_token_ids] = _MINUS_INF


class RepetitiveTokensModifier(NextTokenLogitsModifier):
    """Decreases probability (and logits) for tokens which have already been."""

    def __init__(
            self,
            penalty: float,
            token_ids_to_penalize: torch.tensor
    ):
        """
        Args:
            penalty:
                Repetitive tokens penalization strength (must be >= 1.0).
                1.0 means no penalty.
            token_ids_to_penalize:
                Tensor with shape (batch_size, seq_len).
        Raises:
            NextTokenLogitsModifierError: in case of incorrect input arguments.
        """
        self._token_ids_to_penalize = token_ids_to_penalize
        self._penalty = penalty
        self._check_arguments_validity()

    def _check_arguments_validity(self) -> None:
        if self._penalty < 1.0:
            raise NextTokenLogitsModifierError(
                "`penalty` must be >= 1.0. Use 1.0 if you don't want to apply "
                "repetition penalty.")

    def modify(self, logits: torch.tensor):
        for i in range(logits.size()[0]):
            ids_to_penalize = self._token_ids_to_penalize[i]
            _penalize_logits_tensor(logits[i], ids_to_penalize, self._penalty)


def _penalize_logits_tensor(logits, penalty_idx, penalty):
    if penalty == 1.0:
        return

    idx = torch.unique(penalty_idx)
    logits -= logits.max()

    full_exp = torch.exp(logits)

    e = full_exp[idx]
    sum_e = torch.sum(e)
    s = torch.sum(full_exp) - sum_e

    n = torch.log((e * s) / (penalty * s + penalty * sum_e - sum_e))
    logits[idx] = n
