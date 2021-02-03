import numpy as np


class ArrayMapper:
    """Map an array of values to integer indices ascending from 0, and store this mapping for application to other arrays of floats.

    This code improves on previous approaches (see e.g. https://github.com/brightway-lca/brightway2-calc/blob/114c272a0fde9301ff5e250e2773c01c28d7fd99/bw2calc/indexing.py) by:

    * Not using ``np.searchsorted``, which appears to be always slower than ``index_with_arrays`` (see ``dev`` folder).
    * Keeping the indexing arrays in memory, avoiding their creation each time they are applied to a new matrix.

    .. code-block:: python

        In [1]: from matrix_utils import ArrayMapper
           ...: import numpy as np

        In [2]: am = ArrayMapper(array=np.array([0, 4, 6]))

        In [3]: am.map_array(np.array([6, 6, 3, 0]))
        Out[3]: array([ 2,  2, -1,  0])

    """

    def __init__(self, *, array: np.ndarray, sparse_cutoff: float = 0.1):
        self._check_input_array(array)
        # Even if already unique, this only adds ~2ms for 100.000 elements
        self.array = np.unique(array)

        # TODO
        # Sparse matrices could be used if the number of values present is much less
        # than the number of possible values, given the (min, max) range.
        # The default code will generate a complete mapping for the (min, max)
        # interval, which can use too much memory in certain cases.
        # self.use_sparse = len(self.keys) / self.keys.max() <= sparse_cutoff:

        self.max_value = int(self.array.max())
        self.index_array = np.zeros(self.max_value + 1) - 1
        self.index_array[self.array] = np.arange(len(self.array))

    def __len__(self):
        return self.array.shape[0]

    def _check_input_array(self, array: np.ndarray) -> None:
        if len(array.shape) != 1:
            raise ValueError("array must be 1-d")
        if array.min() < 0:
            raise ValueError("Array index values must be positive")

    def map_array(self, array: np.ndarray) -> np.ndarray:
        self._check_input_array(array)

        result = np.zeros_like(array) - 1
        mask = array <= self.max_value
        result[mask] = self.index_array[array[mask]]
        return result

    def to_dict(self) -> dict:
        """Turn the mapping arrays into a Python dict. This is only useful for
        human examination, the normal implementation uses Numpy functions on the
        arrays directly."""
        return {int(x): int(y) for x, y in zip(self.array, self.map_array(self.array))}
