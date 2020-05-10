from typing import Sequence, TypeVar, Optional, List

TNum = TypeVar('TNum', int, float)


def pad_sequences_from_right(
        sequences: Sequence[Sequence[TNum]],
        max_len: Optional[int],
        pad_value: TNum
) -> List[List[TNum]]:
    max_seq_len = max([len(x) for x in sequences])
    max_len = min(max_len, max_seq_len) if max_len else max_seq_len

    new_seqs = []

    for seq in sequences:
        new_seq = list(seq[:max_len])
        new_seq.extend([pad_value] * (max_len - len(new_seq)))
        new_seqs.append(new_seq)

    return new_seqs
