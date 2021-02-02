from fixtures import basic_mm
from matrix_utils import MappedMatrix
import numpy as np


def test_mappers(basic_mm):
    given = np.array([4, 2, 1, 0])
    expected = np.array([2, 1, -1, 0])
    mm = MappedMatrix(
        packages=[basic_mm], matrix="foo", use_arrays=False, use_distributions=False,
    )
    print("rm ia:", mm.row_mapper.index_array)
    print("rm arr:", mm.row_mapper.array)
    result = mm.row_mapper.map_array(given)
    assert np.allclose(result, expected)


def test_group_filtering(basic_mm):
    mm = MappedMatrix(
        packages=[basic_mm], matrix="foo", use_arrays=False, use_distributions=False,
    )
    assert len(mm.groups) == 2
    assert mm.groups[0].label == "vector"
    assert mm.groups[1].label == "vector2"


def test_indices(basic_mm):
    mm = MappedMatrix(
        packages=[basic_mm], matrix="foo", use_arrays=False, use_distributions=False,
    )
    expected_row = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    expected_col = np.array([0, 1, 2, 3, 7, 6, 5, 4])
    assert len(mm.row_indices) == 2
    assert np.allclose(np.hstack(mm.row_indices), expected_row)
    assert np.allclose(np.hstack(mm.col_indices), expected_col)
