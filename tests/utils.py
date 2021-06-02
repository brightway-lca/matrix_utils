from matrix_utils.errors import AllArraysEmpty
from matrix_utils.utils import safe_concatenate
import numpy as np
import pytest


def test_safe_concatenate():
    a = np.array([0, 1, 2])
    b = np.array([3, 4])
    c = np.array([])
    expected = np.array([0, 1, 2, 3, 4])
    result = safe_concatenate([a, b, c])
    assert np.allclose(expected, result)


def test_safe_concatenate_empty_arrays():
    a = np.array([])
    b = np.array([])
    result = safe_concatenate([a, b])
    assert result.shape == (0,)
    assert isinstance(result, np.ndarray)


def test_safe_concatenate_error():
    with pytest.raises(AllArraysEmpty):
        safe_concatenate([])


def test_safe_concatenate_empty_ok():
    result = safe_concatenate([], empty_ok=True)
    assert result.shape == (0,)
    assert isinstance(result, np.ndarray)
