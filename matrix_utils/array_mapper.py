import numpy as np
from scipy import sparse

from matrix_utils.errors import EmptyArray


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

    """  # NOQA: E501

    def __init__(self, *, array: np.ndarray, sparse_cutoff: int = 50000, empty_ok: bool = False):
        self._check_input_array(array)

        # Even if already unique, this only adds ~2ms for 100.000 elements
        self.array = np.unique(array)
        self.empty_ok = empty_ok

        if self.array.shape == (0,):
            if self.empty_ok:
                self.empty_input = True
                self.max_value = 0
                self.max_index = 0
                return
            else:
                raise EmptyArray("Empty array can't be used to map values")
        else:
            self.empty_input = False
            self.max_value = self.array[-1]
            self.max_index = len(self.array) - 1

        # Zero serves as a missing value, so start at one
        self.matrix = sparse.coo_matrix(
            (np.arange(1, len(self.array) + 1), (self.array, np.zeros_like(self.array))),
            (self.max_value + 1, 1),
        ).tocsc()

    def __len__(self):
        return self.array.shape[0]

    def _check_input_array(self, array: np.ndarray) -> None:
        if len(array.shape) != 1:
            raise ValueError("array must be 1-d")
        if array.shape[0] and array.min() < 0:
            raise ValueError("Array index values must be positive")

    def map_array(self, array: np.ndarray) -> np.ndarray:
        self._check_input_array(array)

        if array.shape == (0,):
            # Empty array
            return array.copy()
        elif self.empty_input:
            if self.empty_ok:
                # Return all with missing flag
                return np.zeros_like(array) - 1
            else:
                raise EmptyArray("Can't map with empty input array")

        result = np.zeros_like(array) - 1
        mask = array <= self.max_value
        # https://numpy.org/doc/stable/reference/generated/numpy.matrix.A1.html
        result[mask] = np.asarray(self.matrix[array[mask], np.zeros_like(array[mask])]).ravel() - 1
        return result

    def to_dict(self) -> dict:
        """Turn the mapping arrays into a Python dict. This is only useful for
        human examination, the normal implementation uses Numpy functions on the
        arrays directly."""
        if self.empty_input:
            return {}
        return {int(x): int(y) for x, y in zip(self.array, self.map_array(self.array))}

    def reverse_dict(self) -> dict:
        return {y: x for x, y in self.to_dict().items()}
