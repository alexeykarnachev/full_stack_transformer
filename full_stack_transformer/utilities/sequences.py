from typing import Sequence, Optional

import numpy as np


def pad_sequences(
        sequences: Sequence[Sequence[int]],
        pad_val: int,
        maxlen: Optional[int] = None
) -> np.ndarray:
    """Pads the sequences.

    Args:
        sequences:
            Collection with 1d sequences.

        maxlen:
            Max sequences length to pad to. If None, the `maxlen` value will
            be equal to the max length of the presented sequences.

        pad_val:
            The padding value.

    Returns:
        2d array of shape (N, M), where N is a number of input sequences and
        M is a max length.
    """
    max_seq_len = max([len(x) for x in sequences])
    maxlen = min(maxlen, max_seq_len) if maxlen else max_seq_len

    new_seqs = []

    for seq in sequences:
        new_seq = list(seq[-maxlen:])
        new_seq.extend([pad_val] * (maxlen - len(new_seq)))
        new_seqs.append(new_seq)

    return np.vstack(new_seqs)
