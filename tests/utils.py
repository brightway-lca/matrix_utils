import bw_processing as bwp
import numpy as np
import pytest

from matrix_utils.errors import AllArraysEmpty
from matrix_utils.utils import has_relevant_data, safe_concatenate_indices


def test_safe_concatenate_indices():
    a = np.array([0, 1, 2])
    b = np.array([3, 4])
    c = np.array([])
    expected = np.array([0, 1, 2, 3, 4])
    result = safe_concatenate_indices([a, b, c])
    assert np.allclose(expected, result)


def test_safe_concatenate_indices_empty_arrays():
    a = np.array([])
    b = np.array([])
    result = safe_concatenate_indices([a, b])
    assert result.shape == (0,)
    assert isinstance(result, np.ndarray)


def test_safe_concatenate_indices_error():
    with pytest.raises(AllArraysEmpty):
        safe_concatenate_indices([])


def test_safe_concatenate_indices_empty_ok():
    result = safe_concatenate_indices([], empty_ok=True)
    assert result.shape == (0,)
    assert isinstance(result, np.ndarray)
    assert result.dtype == int


def create_dp(vector, array, distributions):
    dp = bwp.create_datapackage()
    if distributions:
        dp.add_persistent_vector(
            matrix="foo",
            name="distributions",
            indices_array=np.array([(0, 0)], dtype=bwp.INDICES_DTYPE),
            distributions_array=np.array(
                [
                    (4, 0.5, np.NaN, np.NaN, 0.2, 0.8, False),
                ],
                dtype=bwp.UNCERTAINTY_DTYPE,
            ),
        )
    if vector:
        dp.add_persistent_vector(
            matrix="foo",
            name="vector",
            indices_array=np.array(
                [(10, 10), (12, 9), (14, 8), (18, 7)], dtype=bwp.INDICES_DTYPE
            ),
            data_array=np.array([11, 12.3, 14, 125]),
        )
    if array:
        dp.add_persistent_array(
            matrix="foo",
            name="array",
            indices_array=np.array(
                [(1, 0), (2, 1), (5, 1), (8, 1)], dtype=bwp.INDICES_DTYPE
            ),
            data_array=np.array([[1, 2.3, 4, 25]]).T,
        )
    return dp


def test_has_relevant_data_use_vector():
    assert not has_relevant_data(
        "vector", create_dp(False, False, False), True, False, False
    )
    assert has_relevant_data(
        "vector", create_dp(True, False, False), True, False, False
    )
    assert has_relevant_data("vector", create_dp(True, True, True), True, False, False)
    assert not has_relevant_data(
        "distributions", create_dp(False, False, True), True, False, False
    )
    assert not has_relevant_data(
        "array", create_dp(False, True, False), True, False, False
    )


def test_has_relevant_data_use_distributions():
    assert not has_relevant_data(
        "distributions", create_dp(False, False, False), False, False, True
    )
    assert has_relevant_data(
        "distributions", create_dp(False, False, True), False, False, True
    )
    assert has_relevant_data(
        "distributions", create_dp(True, True, True), False, False, True
    )
    assert has_relevant_data(
        "vector", create_dp(True, False, False), False, False, True
    )
    assert not has_relevant_data(
        "array", create_dp(False, True, False), False, False, True
    )


def test_has_relevant_data_use_array():
    assert not has_relevant_data(
        "array", create_dp(False, False, False), False, True, False
    )
    assert has_relevant_data("array", create_dp(False, True, False), False, True, False)
    assert has_relevant_data("array", create_dp(True, True, True), False, True, False)
    assert not has_relevant_data(
        "vector", create_dp(True, False, False), False, True, False
    )
    assert not has_relevant_data(
        "distributions", create_dp(False, True, False), False, True, False
    )
