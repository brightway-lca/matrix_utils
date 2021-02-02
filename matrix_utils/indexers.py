from numpy.random import RandomState
from typing import Union

# Max signed 32 bit integer, compatible with Windows
MAX_SIGNED_32BIT_INT = 2147483647


class RandomIndexer(RandomState):
    """A (potentially) seeded integer RNG that remembers the generated index.

    Returns indices for a sample array.

    max_value: Number of columns in the array for which a column index is returned.
    seed: Seed for RNG. Optional."""

    def __init__(self, *, max_value: int, seed: Union[int, None] = None):
        self.max_value = max_value
        self.seed_value, self.count, self.index = seed, 0, None
        super().__init__(seed)

    def __next__(self):
        self.index = self.randint(0, MAX_SIGNED_32BIT_INT) % self.ncols
        self.count += 1
        return self.index


class SequentialIndexer:
    def __init__(self, *, max_value: int):
        self.max_value = max_value

    def __next__(self):
        self.index = self.count % self.max_value
        self.count += 1
        return self.index

    def reset_sequential_indices(self):
        """Reset index value.

        Used in Monte Carlo calculations."""
        self.count, self.index = 0, 0


class CombinatorialIndexerMother:
    pass


class CombinatorialIndexerDaughter:
    pass
