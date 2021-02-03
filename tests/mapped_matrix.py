from fixtures import basic_mm
from matrix_utils import MappedMatrix
import numpy as np


def test_mappers():
    given = np.array([4, 2, 1, 0])
    expected = np.array([2, 1, -1, 0])
    mm = MappedMatrix(
        packages=[basic_mm()], matrix="foo", use_arrays=False, use_distributions=False,
    )
    result = mm.row_mapper.map_array(given)
    assert np.allclose(result, expected)


def test_group_filtering():
    mm = MappedMatrix(
        packages=[basic_mm()], matrix="foo", use_arrays=False, use_distributions=False,
    )
    assert len(mm.groups) == 2
    assert mm.groups[0].label == "vector"
    assert mm.groups[1].label == "vector2"
