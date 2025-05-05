import math
from typing import List, Union, Any


def product(probs: Union[None, List[float]]) -> float:
    if probs is None or len(probs) == 0:
        return 0.
    if 0 in probs:
        return 0
    prob = math.exp(sum([math.log(p) for p in probs]))
    return prob


def batchify(input_batch: List[Any], batch_size: int) -> List[List[Any]]:
    l = len(input_batch)
    output = []
    for i in range(0, l, batch_size):
        batch = input_batch[i:min(i + batch_size, l)]
        output.append(batch)
    return output

