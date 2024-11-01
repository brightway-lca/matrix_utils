import bw_processing as bwp
import numpy as np
import pytest
from fixtures import aggregation, basic_mm, diagonal, overlapping

from matrix_utils import MappedMatrix


@pytest.mark.parametrize("smaller", [True, False])
def test_basic_resource_group_indices_dtype(smaller):
    dp = basic_mm(indices_32bit=smaller)

    dtype = np.int32 if smaller else np.int64

    for name in ("vector", "vector2", "array"):
        assert dp.get_resource(f"{name}.indices")[0]["row"].dtype == dtype
        assert dp.get_resource(f"{name}.indices")[0]["col"].dtype == dtype


@pytest.mark.parametrize("smaller", [True, False])
def test_basic_matrix_construction(smaller):
    mm = MappedMatrix(
        packages=[basic_mm(indices_32bit=smaller)],
        matrix="foo",
        use_arrays=False,
        use_distributions=False,
    )
    row = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    col = np.array([0, 1, 2, 3, 7, 6, 5, 4])
    data = np.array([1, 2.3, 4, 25, 11, 12.3, 14, 125])
    matrix = mm.matrix.tocoo()
    assert np.allclose(matrix.row, row)
    assert np.allclose(matrix.col, col)
    assert np.allclose(matrix.data, data)


@pytest.mark.parametrize("smaller", [True, False])
def test_matrix_construction_transpose(smaller):
    mm = MappedMatrix(
        packages=[diagonal(indices_32bit=smaller)],
        matrix="foo",
        use_arrays=False,
        use_distributions=False,
        transpose=True,
    )
    row = np.array([0, 1, 1, 1])
    col = np.array([2, 0, 1, 3])
    data = np.array([4, 1, -2.3, 25])
    assert mm.matrix.shape == (2, 4)
    matrix = mm.matrix.tocoo()
    assert np.allclose(matrix.row, row)
    assert np.allclose(matrix.col, col)
    assert np.allclose(matrix.data, data)

    mm = MappedMatrix(
        packages=[diagonal(indices_32bit=smaller)],
        matrix="foo",
        use_arrays=False,
        use_distributions=False,
    )
    assert mm.matrix.shape == (4, 2)


@pytest.mark.parametrize("smaller", [True, False])
def test_matrix_construction_overlapping_substitution(smaller):
    mm = MappedMatrix(
        packages=[
            overlapping(
                indices_32bit=smaller, sum_intra_duplicates=True, sum_inter_duplicates=False
            )
        ],
        matrix="foo",
        use_arrays=False,
        use_distributions=False,
    )
    mm.rebuild_matrix()
    row = np.array([0, 1, 2, 3, 4, 5])
    col = np.array([0, 1, 2, 3, 5, 4])
    data = np.array([11, 14, 4, 25, 12.3, 125])
    matrix = mm.matrix.tocoo()
    assert np.allclose(matrix.row, row)
    assert np.allclose(matrix.col, col)
    assert np.allclose(matrix.data, data)


@pytest.mark.parametrize("smaller", [True, False])
def test_matrix_construction_overlapping_sum(smaller):
    mm = MappedMatrix(
        packages=[
            overlapping(
                indices_32bit=smaller, sum_intra_duplicates=False, sum_inter_duplicates=True
            )
        ],
        matrix="foo",
        use_arrays=False,
        use_distributions=False,
    )
    mm.rebuild_matrix()
    row = np.array([0, 1, 2, 3, 4, 5])
    col = np.array([0, 1, 2, 3, 5, 4])
    data = np.array([12, 16.3, 4, 25, 12.3, 125])
    matrix = mm.matrix.tocoo()
    assert np.allclose(matrix.row, row)
    assert np.allclose(matrix.col, col)
    assert np.allclose(matrix.data, data)


@pytest.mark.parametrize("smaller", [True, False])
def test_matrix_construction_internal_aggregation(smaller):
    mm = MappedMatrix(
        packages=[aggregation(indices_32bit=smaller)],
        matrix="foo",
        use_arrays=False,
        use_distributions=False,
    )
    row = np.array([0, 1, 2, 3])
    col = np.array([0, 1, 2, 3])
    data = np.array([1, 2.3, 21, 25])
    matrix = mm.matrix.tocoo()
    assert np.allclose(matrix.row, row)
    assert np.allclose(matrix.col, col)
    assert np.allclose(matrix.data, data)


@pytest.mark.parametrize("smaller", [True, False])
def test_matrix_construction_no_internal_aggregation(smaller):
    mm = MappedMatrix(
        packages=[aggregation(indices_32bit=smaller, sum_intra_duplicates=False)],
        matrix="foo",
        use_arrays=False,
        use_distributions=False,
    )
    row = np.array([0, 1, 2, 3])
    col = np.array([0, 1, 2, 3])
    data = np.array([1, 2.3, 17, 25])
    matrix = mm.matrix.tocoo()
    assert np.allclose(matrix.row, row)
    assert np.allclose(matrix.col, col)
    assert np.allclose(matrix.data, data)


def test_arrays_sequential_iteration():
    dp = bwp.create_datapackage()
    dp.add_persistent_vector(
        matrix="foo",
        indices_array=np.array(
            [
                (100, 400),
                (101, 401),
                (102, 402),
                (103, 403),  # Production
                (100, 401),
                (101, 402),
                (101, 403),
                (102, 403),
            ],  # Inputs
            dtype=bwp.INDICES_DTYPE,  # Means first element is row, second is column
        ),
        flip_array=np.array(
            [False, False, False, False, True, True, True, True]  # Production  # Inputs
        ),
        data_array=np.array([1, 1, 1, 1, 2, 4, 8, 16]),  # Production  # Inputs
    )
    s = bwp.create_datapackage(sequential=True)
    s.add_persistent_array(
        matrix="foo",
        data_array=np.array(
            [
                [-10, -6],  # Amount of 101 needed by 404
                [-6, -10],  # Amount of 101 needed by 405
                [0, -20],  # Amount of 102 needed by 404
                [-20, 0],  # Amount of 102 needed by 405
                [1, 1],  # Production of 404
                [1, 1],  # Production of 405
            ]
        ),
        indices_array=np.array(
            [(101, 404), (101, 405), (102, 404), (102, 405), (104, 404), (105, 405)],
            dtype=bwp.INDICES_DTYPE,
        ),
    )
    mm = MappedMatrix(
        packages=[dp, s],
        matrix="foo",
        use_arrays=True,
        use_distributions=False,
    )
    assert mm.matrix[1, 4] == -10
    next(mm)
    assert mm.matrix[1, 4] == -6
