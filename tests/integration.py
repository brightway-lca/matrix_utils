from fixtures import overlapping, basic_mm, aggregation
from matrix_utils import MappedMatrix
import numpy as np


def test_basic_matrix_construction():
    mm = MappedMatrix(
        packages=[basic_mm()], matrix="foo", use_arrays=False, use_distributions=False,
    )
    row = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    col = np.array([0, 1, 2, 3, 7, 6, 5, 4])
    data = np.array([1, 2.3, 4, 25, 11, 12.3, 14, 125])
    matrix = mm.matrix.tocoo()
    assert np.allclose(matrix.row, row)
    assert np.allclose(matrix.col, col)
    assert np.allclose(matrix.data, data)


def test_matrix_construction_overlapping_substitution():
    mm = MappedMatrix(
        packages=[overlapping(sum_inter_duplicates=True)], matrix="foo", use_arrays=False, use_distributions=False,
    )
    mm.rebuild_matrix()
    row = np.array([0, 1, 2, 3, 4, 5])
    col = np.array([0, 1, 2, 3, 5, 4])
    data = np.array([11, 14, 4, 25, 12.3, 125])
    matrix = mm.matrix.tocoo()
    assert np.allclose(matrix.row, row)
    assert np.allclose(matrix.col, col)
    assert np.allclose(matrix.data, data)


def test_matrix_construction_overlapping_sum():
    mm = MappedMatrix(
        packages=[overlapping(sum_inter_duplicates=False)], matrix="foo", use_arrays=False, use_distributions=False,
    )
    mm.rebuild_matrix()
    row = np.array([0, 1, 2, 3, 4, 5])
    col = np.array([0, 1, 2, 3, 5, 4])
    data = np.array([12, 16.3, 4, 25, 12.3, 125])
    matrix = mm.matrix.tocoo()
    assert np.allclose(matrix.row, row)
    assert np.allclose(matrix.col, col)
    assert np.allclose(matrix.data, data)


def test_matrix_construction_internal_aggregation():
    mm = MappedMatrix(
        packages=[aggregation()], matrix="foo", use_arrays=False, use_distributions=False,
    )
    row = np.array([0, 1, 2, 3])
    col = np.array([0, 1, 2, 3])
    data = np.array([1, 2.3, 21, 25])
    matrix = mm.matrix.tocoo()
    assert np.allclose(matrix.row, row)
    assert np.allclose(matrix.col, col)
    assert np.allclose(matrix.data, data)


def test_matrix_construction_no_internal_aggregation():
    mm = MappedMatrix(
        packages=[aggregation(sum_intra_duplicates=False)], matrix="foo", use_arrays=False, use_distributions=False,
    )
    row = np.array([0, 1, 2, 3])
    col = np.array([0, 1, 2, 3])
    data = np.array([1, 2.3, 17, 25])
    matrix = mm.matrix.tocoo()
    assert np.allclose(matrix.row, row)
    assert np.allclose(matrix.col, col)
    assert np.allclose(matrix.data, data)
