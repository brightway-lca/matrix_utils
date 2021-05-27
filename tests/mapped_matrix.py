from fixtures import basic_mm, diagonal
from matrix_utils import MappedMatrix
from matrix_utils.errors import EmptyArray
import numpy as np
import pytest


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


def test_no_packages():
    pass


def test_no_useful_pacakges():
    pass


def test_diagonal_matrix():
    mm = MappedMatrix(
        packages=[diagonal()], matrix="foo", use_arrays=False, use_distributions=False,
    )
    assert mm.matrix.shape == (4, 2)
    assert np.allclose(mm.matrix.data, [1, -2.3, 4, 25])

    mm = MappedMatrix(
        packages=[diagonal()], matrix="foo", use_arrays=False, use_distributions=False, diagonal=True,
    )
    assert mm.matrix.shape == (4, 4)
    for x, y in zip(range(4), [1, -2.3, 4, 25]):
        assert mm.matrix[x, x] == y

    assert np.allclose(mm.matrix.data, [1, -2.3, 4, 25])


def test_custom_filter():
    mm = MappedMatrix(
        packages=[diagonal()], matrix="foo", use_arrays=False, use_distributions=False,
    )
    assert mm.matrix.shape == (4, 2)

    mm = MappedMatrix(
        packages=[diagonal()], matrix="foo", use_arrays=False, use_distributions=False, custom_filter=lambda x: x['col'] == 1
    )
    assert mm.matrix.shape == (3, 1)
    assert mm.matrix.sum() == 1 - 2.3 + 25

    with pytest.raises(EmptyArray):
        mm = MappedMatrix(
            packages=[diagonal()], matrix="foo", use_arrays=False, use_distributions=False, custom_filter=lambda x: x['col'] == 2
        )
