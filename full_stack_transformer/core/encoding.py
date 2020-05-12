from dataclasses import dataclass
from typing import List


@dataclass
class Encoding:
    token_ids: List[int]
    lm_labels: List[int]