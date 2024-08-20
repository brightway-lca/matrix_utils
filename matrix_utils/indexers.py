from itertools import product
from typing import List, Union

from numpy import int32, int64
from numpy.random import PCG64, Generator

# Max signed 32 bit integer, compatible with Windows
MAX_SIGNED_32BIT_INT = 2147483647
MAX_SIGNED_64BIT_INT = 9223372036854775807


class Indexer:
    """Base class for indexers"""

    pass


class RandomIndexer(Generator, Indexer):
    """A (potentially) seeded integer RNG that remembers the generated index.

    Returns indices for a sample array.

    Because we are creating Python indices, and Python starts indices with zero, the value
    `max_value` is never reached - the maximum generated index is `max_value - 1`.

    max_value: Number of columns in the array for which a column index is returned.
    seed: Seed for RNG. Optional."""

    def __init__(self, seed: Union[int, None] = None, max_value: int = MAX_SIGNED_64BIT_INT):
        self.seed = seed
        self.max_value = max_value
        self.reset()

    def __next__(self):
        self.index = self.integers(
            0, self.max_value, dtype=int32 if self.max_value == MAX_SIGNED_32BIT_INT else int64
        )
        return self.index

    def reset(self):
        super().__init__(PCG64(self.seed))
        next(self)


class SequentialIndexer(Indexer):
    def __init__(self, offset: int = 0):
        self.reset(offset=offset)

    def __next__(self):
        self.index += 1
        return self.index

    def reset(self, offset=0):
        self.index = offset


class CombinatorialIndexer(Indexer):
    def __init__(self, max_values: List[int]):
        self.max_values = max_values
        self.reset()

    def __next__(self):
        self.index = next(self.iterator)
        return self.index

    def reset(self):
        self.iterator = product(*[range(x) for x in self.max_values])
        next(self)


class Proxy:
    """Simple class that proxies access to a combinatorial indexer"""

    def __init__(self, indexer: CombinatorialIndexer, offset: int):
        self.indexer = indexer
        self.offset = offset

    @property
    def index(self):
        return self.indexer.index[self.offset]

    def reset(self):
        self.indexer.reset()
