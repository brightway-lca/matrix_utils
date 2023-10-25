import bw_processing as bwp
import numpy as np
import pytest

from matrix_utils import ArrayMapper, MappedMatrix, MappedMatrixDict
from matrix_utils.errors import AllArraysEmpty
from matrix_utils.indexers import RandomIndexer, SequentialIndexer


@pytest.fixture
def mmd_fixture():
    first = bwp.create_datapackage()
    first.add_persistent_vector(
        matrix="foo",
        name="vector",
        indices_array=np.array([(0, 0), (2, 1)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([1, 2.3]),
    )
    second = bwp.create_datapackage()
    second.add_persistent_vector(
        matrix="foo",
        name="vector",
        indices_array=np.array([(10, 10), (12, 11)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([11, 12]),
    )
    third = bwp.create_datapackage()
    third.add_persistent_vector(
        matrix="foo",
        name="vector",
        indices_array=np.array([(0, 10), (2, 11)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([11, 12]),
    )
    fourth = bwp.create_datapackage()
    fourth.add_persistent_array(
        matrix="foo",
        name="array",
        indices_array=np.array(
            [(1, 0), (2, 1), (5, 1), (8, 1)], dtype=bwp.INDICES_DTYPE
        ),
        data_array=np.arange(8).reshape((4, 2)) + 10,
    )
    fifth = bwp.create_datapackage()
    fifth.add_persistent_array(
        matrix="foo",
        name="array",
        indices_array=np.array(
            [(1, 0), (12, 11), (5, 1), (18, 11)], dtype=bwp.INDICES_DTYPE
        ),
        data_array=np.arange(20).reshape((4, 5)),
    )
    row_mapper = ArrayMapper(
        array=np.array([0, 2, 10, 12, 0, 2, 1, 2, 5, 8, 1, 12, 5, 18])
    )
    col_mapper = ArrayMapper(
        array=np.array([0, 1, 10, 11, 10, 11, 0, 1, 1, 1, 0, 11, 1, 11])
    )
    return first, second, third, fourth, fifth, row_mapper, col_mapper


def test_basic_mmd_construction(mmd_fixture):
    first, second, third, fourth, fifth, rows, cols = mmd_fixture
    mmd = MappedMatrixDict(
        packages={"a": [first, second], "b": [third, fourth], "c": [fifth]},
        matrix="foo",
        row_mapper=rows,
        col_mapper=cols,
        use_arrays=True,
    )

    assert len(mmd) == 3
    assert len(mmd.matrices) == 3
    for key in mmd:
        mm = mmd[key]
        assert isinstance(mm, MappedMatrix)
        assert mm.matrix.shape == (8, 4)


def test_basic_mmd_as_dict(mmd_fixture):
    first, second, third, fourth, fifth, rows, cols = mmd_fixture
    mmd = MappedMatrixDict(
        packages={"a": [first, second], "b": [third, fourth], "c": [fifth]},
        matrix="foo",
        row_mapper=rows,
        col_mapper=cols,
        use_arrays=True,
    )

    assert "a" in mmd
    assert "g" not in mmd
    assert len(mmd) == 3
    assert mmd.keys()
    assert list(mmd.keys()) == ["a", "b", "c"]
    assert mmd.values()
    with pytest.raises(TypeError):
        del mmd["a"]


def test_mmd_shared_indexer(mmd_fixture):
    first, second, third, fourth, fifth, rows, cols = mmd_fixture
    mmd = MappedMatrixDict(
        packages={"a": [first, second], "b": [third, fourth], "c": [fifth]},
        matrix="foo",
        row_mapper=rows,
        col_mapper=cols,
        use_arrays=True,
    )

    for mm in mmd.values():
        for group in mm.groups:
            assert group.indexer is mmd.global_indexer


def test_mmd_iterate_indexer_changes_matrix_values(mmd_fixture):
    first, second, third, fourth, fifth, rows, cols = mmd_fixture
    mmd = MappedMatrixDict(
        packages={"a": [first, second], "b": [third, fourth], "c": [fifth]},
        matrix="foo",
        row_mapper=rows,
        col_mapper=cols,
        use_arrays=True,
        sequential=True,
    )

    for mm, value in zip(
        mmd.values(), [1 + 2.3 + 11 + 12, 11 + 12 + 10 + 12 + 14 + 16, 0 + 5 + 10 + 15]
    ):
        assert mm.matrix.sum() == value

    next(mmd)

    for mm, value in zip(
        mmd.values(), [1 + 2.3 + 11 + 12, 11 + 12 + 11 + 13 + 15 + 17, 1 + 6 + 11 + 16]
    ):
        assert mm.matrix.sum() == value


def test_mmd_empty_datapackages(mmd_fixture):
    first, second, third, fourth, fifth, rows, cols = mmd_fixture
    MappedMatrixDict(
        packages={"a": [first, second], "b": [third, fourth], "c": [fifth]},
        matrix="foo",
        row_mapper=rows,
        col_mapper=cols,
    )

    with pytest.raises(AllArraysEmpty):
        MappedMatrixDict(
            packages={
                "a": [],
                "b": [],
            },
            matrix="foo",
            row_mapper=rows,
            col_mapper=cols,
        )

    MappedMatrixDict(
        packages={
            "a": [],
            "b": [],
        },
        matrix="foo",
        row_mapper=rows,
        col_mapper=cols,
        empty_ok=True,
    )


def test_mmd_random(mmd_fixture):
    first, second, third, fourth, fifth, rows, cols = mmd_fixture
    indexer = RandomIndexer(seed=42)
    mmd = MappedMatrixDict(
        packages={"a": [first, second], "b": [third, fourth], "c": [fifth]},
        matrix="foo",
        row_mapper=rows,
        col_mapper=cols,
        use_arrays=True,
        indexer_override=indexer,
    )

    for mm in mmd.values():
        for group in mm.groups:
            assert group.indexer is indexer
            assert group.indexer.index == 191664963

    next(mmd)

    for mm in mmd.values():
        for group in mm.groups:
            assert group.indexer is indexer
            assert group.indexer.index == 1662057957


def test_mmd_custom_indexer(mmd_fixture):
    first, second, third, fourth, fifth, rows, cols = mmd_fixture
    indexer = SequentialIndexer()
    mmd = MappedMatrixDict(
        packages={"a": [first, second], "b": [third, fourth], "c": [fifth]},
        matrix="foo",
        row_mapper=rows,
        col_mapper=cols,
        use_arrays=True,
        indexer_override=indexer,
    )

    for mm in mmd.values():
        for group in mm.groups:
            assert group.indexer is indexer
            assert group.indexer.index == 0

    next(indexer)

    for mm in mmd.values():
        for group in mm.groups:
            assert group.indexer is indexer
            assert group.indexer.index == 1


def test_mmd_sequential(mmd_fixture):
    first, second, third, fourth, fifth, rows, cols = mmd_fixture
    mmd = MappedMatrixDict(
        packages={"a": [first, second], "b": [third, fourth], "c": [fifth]},
        matrix="foo",
        row_mapper=rows,
        col_mapper=cols,
        use_arrays=True,
        sequential=True,
    )

    for mm in mmd.values():
        for group in mm.groups:
            assert group.indexer.index == 0

    next(mmd)

    for mm in mmd.values():
        for group in mm.groups:
            assert group.indexer.index == 1


def test_mmd_invalid_packages(mmd_fixture):
    first, second, third, fourth, fifth, rows, cols = mmd_fixture
    with pytest.raises(ValueError):
        MappedMatrixDict(
            packages=[("a", [first, second]), ("b", [third, fourth]), ("c", [fifth])],
            matrix="foo",
            row_mapper=rows,
            col_mapper=cols,
            use_arrays=True,
        )
