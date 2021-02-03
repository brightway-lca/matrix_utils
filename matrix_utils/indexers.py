from itertools import product
from numpy.random import Generator, PCG64
from typing import Union, List


# Max signed 32 bit integer, compatible with Windows
MAX_SIGNED_32BIT_INT = 2147483647


class Indexer:
    """Base class for indexers"""
    pass


class RandomIndexer(Generator, Indexer):
    """A (potentially) seeded integer RNG that remembers the generated index.

    Returns indices for a sample array.

    max_value: Number of columns in the array for which a column index is returned.
    seed: Seed for RNG. Optional."""

    def __init__(self, seed: Union[int, None] = None):
        super().__init__(PCG64(seed))
        next(self)

    def __next__(self):
        self.index = self.integers(0, MAX_SIGNED_32BIT_INT)
        return self.index


class SequentialIndexer(Indexer):
    def __init__(self):
        self.index = 0

    def __next__(self):
        self.index += 1
        return self.index


class CombinatorialIndexer(Indexer):
    def __init__(self, max_values: List[int]):
        self.iterator = product(*[range(x) for x in max_values])
        next(self)

    def __next__(self):
        self.index = next(self.iterator)
        return self.index


class Proxy:
    """Simple class that proxies access to a combinatorial indexer"""

    def __init__(self, indexer: CombinatorialIndexer, offset: int):
        self.indexer = indexer
        self.offset = offset

    @property
    def index(self):
        return self.indexer[self.offset]
