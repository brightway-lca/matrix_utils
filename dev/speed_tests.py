# -*- coding: utf-8 -*-
import itertools
import numpy as np


def index_with_arrays(array_from, array_to, mapping):
    """Map ``array_from`` keys to ``array_to`` values using the dictionary ``mapping``.

    Turns the keys and values of mapping into index arrays.

    This is needed to take the ``flow``, ``input``, and ``output`` columns, which can be arbitrarily large integers, and transform them to matrix indices, which start from zero.

    Here is an example:

    .. code-block:: python

        import numpy as np
        a_f = np.array((1, 2, 3, 4))
        a_t = np.zeros(4)
        mapping = {1: 5, 2: 6, 3: 7, 4: 8}
        index_with_arrays(a_f, a_t, mapping)
        # => a_t is now [5, 6, 7, 8]

    Args:
        * *array_from* (array): 1-dimensional integer numpy array.
        * *array_to* (array): 1-dimensional integer numpy array.
        * *mapping* (dict): Dictionary that links ``mapping`` indices to ``row`` or ``col`` indices, e.g. ``{34: 3}``.

    Operates in place. Doesn't return anything."""
    keys = np.array(list(mapping.keys()))
    values = np.array(list(mapping.values()))

    if keys.min() < 0:
        raise ValueError("Keys must be positive integers")

    index_array = np.zeros(keys.max() + 1) - 1
    index_array[keys] = values

    mask = array_from <= keys.max()
    array_to[:] = -1
    array_to[mask] = index_array[array_from[mask]]
    # array_to[array_to == -1] = np.nan


def index_with_indexarray(array_from):
    # Twice as fast as index_with_searchsorted
    unique = np.unique(array_from)
    values = np.arange(unique.max() + 1)

    index_array = np.zeros_like(unique) - 1
    index_array[unique] = values

    return index_array[array_from]


def index_with_searchsorted(array_from, array_to=None):
    """Build a dictionary from the sorted, unique elements of an array, and map this dictionary from ``array_from`` to ``array_to``.

    Adapted from http://stackoverflow.com/questions/3403973/fast-replacement-of-values-in-a-numpy-array.

    Here is an example:

    .. code-block:: python

        import numpy as np
        array = np.array((4, 8, 6, 2, 4))
        output = np.zeros(5)
        index_with_searchsorted(array, output)
        # => returns {2: 0, 4: 1, 6: 2, 8: 3}
        # and `output` is [1, 3, 2, 0, 1]

    ``array_from`` and ``array_to`` are arrays of integers.

    Returns a dictionary that maps the sorted, unique elements of ``array_from`` to integers starting with zero."""
    unique = np.unique(array_from)
    idx = np.searchsorted(unique, array_from)
    if array_to is not None:
        array_to[:] = idx
    else:
        array_to = idx
    return array_to, dict(zip((int(x) for x in unique), itertools.count()))


def input_array(num_elements=250_000, num_distinct=20_000):
    return np.random.randint(low=0, high=num_distinct, size=num_elements)


arr = input_array()

%timeit index_with_searchsorted(arr)
%timeit index_with_searchsorted(arr, np.zeros_like(arr))

_, mapping = index_with_searchsorted(arr)
arr2 = input_array()
%timeit index_with_arrays(arr2, np.zeros_like(arr2), mapping)



MAX = 10000
ELEMENTS = 250000

from scipy import sparse

indices_row_int = np.random.randint(low=0, high=MAX, size=ELEMENTS)
indices_row_float = indices_int.astype(np.float32)

indices_col_int = np.random.randint(low=0, high=MAX, size=ELEMENTS)
indices_col_float = indices_int.astype(np.float32)

data = np.random.random(size=(250_000,))

%timeit sparse.coo_matrix((data, (indices_row_int, indices_col_int)), (MAX, MAX))
%timeit sparse.coo_matrix((data, (indices_row_float, indices_col_float)), (MAX, MAX))
