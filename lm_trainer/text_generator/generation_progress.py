import torch


class GenerationProgressError(Exception):
    pass


def _return_value_if_not_initialized(value):
    def _return_value_if_not_initialized_inner(function_or_prop):
        def decorated(self, *args, **kwargs):
            if self._number_of_samples is None:
                return value
            else:
                return function_or_prop(self, *args, **kwargs)

        return decorated

    return _return_value_if_not_initialized_inner


class GenerationProgress:
    """Tracks generation progress."""

    @property
    @_return_value_if_not_initialized(value=False)
    def max_length_reached(self):
        """Will be True, when all sequence lengths will reach max_length."""
        return self.current_length >= self._max_length

    @property
    @_return_value_if_not_initialized(value=False)
    def all_samples_finished(self):
        """Will be True, when eos_token_id will appear in every sequence."""
        return self._unfinished_samples_mask.max() == 0

    @property
    @_return_value_if_not_initialized(value=False)
    def finished(self):
        return self.max_length_reached or self.all_samples_finished

    def __init__(self, eos_token_id: int, max_length: int):
        """
        Args:
            eos_token_id:
                End of string token id. It's needed for GeneratorProgress to
                understand which sample is finished.

            max_length:
                Maximum length of the generated sequences.
        """
        self._eos_token_id = eos_token_id
        self._max_length = max_length
        self.current_length = 0

        self._number_of_samples = None
        self._unfinished_samples_mask = None
        self.generated_sample_lengths = None

        self._check_arguments_validity()

    def _check_arguments_validity(self) -> None:
        if self._max_length < 1:
            raise GenerationProgressError("`max_length` must be >= 1.")
        elif self._eos_token_id < 0:
            raise GenerationProgressError("`eos_token_id` must be >= 0.")

    def update(self, next_token_ids) -> None:
        """Updates generation progress status."""
        self._assert_update_is_possible()
        self._initialize_if_needed(next_token_ids)

        not_eos_tokens_mask = next_token_ids.ne(self._eos_token_id).bool()
        self.generated_sample_lengths[self._unfinished_samples_mask] += 1

        self._unfinished_samples_mask *= not_eos_tokens_mask
        self.current_length += 1

    def _assert_update_is_possible(self):
        if self.finished:
            raise GenerationProgressError(
                "Can't update generation progress, because it's already "
                "finished.")

    def _initialize_if_needed(self, next_tokens):
        if self._number_of_samples is None:
            device = next_tokens.device
            self._number_of_samples = len(next_tokens)
            self._unfinished_samples_mask = torch.ones(
                self._number_of_samples).bool().to(device)
            self.generated_sample_lengths = torch.zeros(
                self._number_of_samples).long().to(device)
