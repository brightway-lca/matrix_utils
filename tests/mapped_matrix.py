from fixtures import basic_mm, diagonal
from matrix_utils import MappedMatrix
from matrix_utils.errors import EmptyArray, AllArraysEmpty
import bw_processing as bwp
import numpy as np
import pytest


def test_mappers():
    given = np.array([4, 2, 1, 0])
    expected = np.array([2, 1, -1, 0])
    mm = MappedMatrix(
        packages=[basic_mm()],
        matrix="foo",
        use_arrays=False,
        use_distributions=False,
    )
    result = mm.row_mapper.map_array(given)
    assert np.allclose(result, expected)


def test_group_filtering():
    mm = MappedMatrix(
        packages=[basic_mm()],
        matrix="foo",
        use_arrays=False,
        use_distributions=False,
    )
    assert len(mm.groups) == 2
    assert mm.groups[0].label == "vector"
    assert mm.groups[1].label == "vector2"


def test_no_useful_packages():
    pass


def test_diagonal_matrix():
    mm = MappedMatrix(
        packages=[diagonal()],
        matrix="foo",
        use_arrays=False,
        use_distributions=False,
    )
    assert mm.matrix.shape == (4, 2)
    assert np.allclose(mm.matrix.data, [1, -2.3, 4, 25])

    mm = MappedMatrix(
        packages=[diagonal()],
        matrix="foo",
        use_arrays=False,
        use_distributions=False,
        diagonal=True,
    )
    assert mm.matrix.shape == (4, 4)
    for x, y in zip(range(4), [1, -2.3, 4, 25]):
        assert mm.matrix[x, x] == y

    assert np.allclose(mm.matrix.data, [1, -2.3, 4, 25])


def test_custom_filter():
    mm = MappedMatrix(
        packages=[diagonal()],
        matrix="foo",
        use_arrays=False,
        use_distributions=False,
    )
    assert mm.matrix.shape == (4, 2)

    mm = MappedMatrix(
        packages=[diagonal()],
        matrix="foo",
        use_arrays=False,
        use_distributions=False,
        custom_filter=lambda x: x["col"] == 1,
    )
    assert mm.matrix.shape == (3, 1)
    assert mm.matrix.sum() == 1 - 2.3 + 25

    with pytest.raises(EmptyArray):
        mm = MappedMatrix(
            packages=[diagonal()],
            matrix="foo",
            use_arrays=False,
            use_distributions=False,
            custom_filter=lambda x: x["col"] == 2,
        )


def test_indexer_override():
    s = bwp.create_datapackage(sequential=True)
    s.add_persistent_array(
        matrix="foo",
        data_array=np.arange(12).reshape(3, 4),
        indices_array=np.array([(0, 0), (1, 1), (0, 1)], dtype=bwp.INDICES_DTYPE),
    )
    mm = MappedMatrix(
        packages=[s],
        matrix="foo",
        use_arrays=True,
        use_distributions=False,
    )
    assert np.allclose(mm.matrix.toarray(), [[0, 8], [0, 4]])
    next(mm)
    assert np.allclose(mm.matrix.toarray(), [[1, 9], [0, 5]])
    next(mm)
    assert np.allclose(mm.matrix.toarray(), [[2, 10], [0, 6]])

    class MyIndexer:
        index = 2

        def __next__(self):
            pass

    mm = MappedMatrix(
        packages=[s],
        matrix="foo",
        use_arrays=True,
        use_distributions=False,
        indexer_override=MyIndexer(),
    )
    assert np.allclose(mm.matrix.toarray(), [[2, 10], [0, 6]])
    next(mm)
    assert np.allclose(mm.matrix.toarray(), [[2, 10], [0, 6]])
    next(mm)
    assert np.allclose(mm.matrix.toarray(), [[2, 10], [0, 6]])


def test_no_packages_error():
    with pytest.raises(AllArraysEmpty):
        MappedMatrix(
            packages=[],
            matrix="foo",
        )


def test_no_packages_empty_ok():
    mm = MappedMatrix(packages=[], matrix="foo", empty_ok=True)
    assert mm.matrix.shape == (0, 0)


def test_no_useful_packages_empty_ok():
    mm = MappedMatrix(packages=[diagonal()], matrix="bar", empty_ok=True)
    assert mm.matrix.shape == (0, 0)


def test_existing_indexer():
    class MyIndexer:
        index = 2

        def __next__(self):
            pass

    s = bwp.create_datapackage(sequential=True)
    s.add_persistent_array(
        matrix="foo",
        data_array=np.arange(12).reshape(3, 4),
        indices_array=np.array([(0, 0), (1, 1), (0, 1)], dtype=bwp.INDICES_DTYPE),
    )
    s.indexer = MyIndexer()
    mm = MappedMatrix(
        packages=[s],
        matrix="foo",
        use_arrays=True,
        use_distributions=False,
    )
    assert np.allclose(mm.matrix.toarray(), [[2, 10], [0, 6]])
    next(mm)
    assert np.allclose(mm.matrix.toarray(), [[2, 10], [0, 6]])
