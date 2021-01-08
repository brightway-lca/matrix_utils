import numpy as np
from scipy import sparse


class ArrayMapper:
    """Map an array of floats to integers ascending from 0, an d store this mapping for application to other arrays of floats.

    This code improves on previous approaches (see e.g. https://github.com/brightway-lca/brightway2-calc/blob/114c272a0fde9301ff5e250e2773c01c28d7fd99/bw2calc/indexing.py) by:

    * Not using ``np.searchsorted``, which appears to be always slower than ``index_with_arrays`` (see ``dev`` folder).
    * Using float arrays as input values with float dtypes, which is somehow faster than integers, and allows for the use of ``NaN`` instead of magic sentinel values.
    * Keeping the indexing arrays in memory, avoiding their creation each time they are applied to a new matrix.

    The standard approach assumes that the provided index values are relatively small (i.e. less than 1 million). In this case, we can flip the normally indexing routine:

    .. code-block:: python

        input_array = np.array([0, 4, 6])
        index_array = np.zeros(input_array.max() + 1) - 1
        index_array[input_array] = np.arange(len(input_array))
        index_array
        >>> array([ 0., -1., -1., -1.,  1., -1.,  2.])

        new_array_to_index = np.array([0, 2, 4])
        index_array[new_array_to_index]
        >>> array([ 0., -1.,  1.])

    We use a sentinel value of -1 to indicate a missing match.

    """

    def __init__(self, *, array: np.ndarray, sparse_cutoff: float = 0.1):
        self._check_input_array(array)
        # Even if already unique, this only adds ~2ms for 100.000 elements
        self.array = np.unique(array)

        # TODO
        # self.use_sparse = len(self.keys) / self.keys.max() <= sparse_cutoff:
        self.use_sparse = False

        self.max_value = self.array.max()
        self.index_array = np.zeros(self.max_value + 1) - 1
        self.index_array[self.array] = np.arange(len(self.array))

    def _check_input_array(self, array: np.ndarray):
        if len(array.shape) != 1:
            raise ValueError("array must be 1-d")
        if array.min() < 0:
            raise ValueError("Array index values must be positive")

    def map_array(self, array):
        self._check_input_array(array)

        result = np.zeros_like(array, dtype=np.float32) - 1
        mask = array <= self.max_value
        result[mask] = self.index_array[array[mask]]

        mask = result == -1
        result[mask] = np.nan
        return result

    def to_dict(self):
        return {int(x): int(y) for x, y in zip(self.array, self.map_array(self.array))}
