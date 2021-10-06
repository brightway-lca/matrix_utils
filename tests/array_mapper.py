import numpy as np
import pytest

from matrix_utils.array_mapper import ArrayMapper
from matrix_utils.errors import EmptyArray


def test_initial_setup():
    inpt = np.array([1, 2, 3, 6, 9, 12, 9, 6, 5])
    am = ArrayMapper(array=inpt)
    expected = np.array([0, 1, 2, 4, 5, 6, 5, 4, 3])
    assert np.allclose(am.map_array(inpt), expected)


def test_float_indices_raises_error():
    inpt = np.array([1, 2, 3, 6.0])
    with pytest.raises(IndexError):
        ArrayMapper(array=inpt)


def test_conversion_to_dict():
    inpt = np.array([1, 2, 3, 6, 9, 12, 9, 6, 5])
    am = ArrayMapper(array=inpt)
    expected = {1: 0, 2: 1, 3: 2, 5: 3, 6: 4, 9: 5, 12: 6}
    assert am.to_dict() == expected


def test_negative_error_instantiation():
    inpt = np.array([1, 2, 3, 6, -9, 12, 9, 6, 5])
    with pytest.raises(ValueError):
        ArrayMapper(array=inpt)


def test_negative_error_mapping():
    inpt = np.array([1, 2, 3, 6, 9, 12, 9, 6, 5])
    inpt2 = np.array([1, 2, 3, 6, -9, 12, 9, 6, 5])
    am = ArrayMapper(array=inpt)
    with pytest.raises(ValueError):
        am.map_array(inpt2)


def test_one_dimensional_input():
    inpt = np.array([[1, 2, 3, 6, 9], [12, 9, 6, 5, 1]])
    with pytest.raises(ValueError):
        ArrayMapper(array=inpt)


def test_mapping_missing_values():
    inpt = np.array([1, 2, 3, 6, 9, 12, 9, 6, 5])
    am = ArrayMapper(array=inpt)
    given = np.array([1, 3, 2, 4])
    result = am.map_array(given)
    assert np.allclose(result, [0, 2, 1, -1])


def test_mapping_out_of_bounds():
    inpt = np.array([1, 2, 3, 6, 9, 12, 9, 6, 5])
    am = ArrayMapper(array=inpt)
    given = np.array([1, 3, 2, 400])
    result = am.map_array(given)
    assert np.allclose(result, [0, 2, 1, -1])


def test_empty_array_error():
    with pytest.raises(EmptyArray):
        ArrayMapper(array=np.array([], dtype=int))


def test_empty_array_ok():
    am = ArrayMapper(array=np.array([], dtype=int), empty_ok=True)
    assert am.array.shape == (0,)
    assert am.index_array.shape == (1,)
