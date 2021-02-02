from scipy import sparse
import numpy as np


def aggregate_with_sparse(
    rows: np.ndarray, cols: np.ndarray, data: np.ndarray, count: int
):
    # About three times faster than pandas groupby for our example use case (100.000 data points, 500 possible row and column indices)
    # See dev/speed_test.py
    # and https://jakevdp.github.io/PythonDataScienceHandbook/03.08-aggregation-and-grouping.html for other use cases

    # Note that the conversion to CSR is necessary for aggregation
    matrix = sparse.coo_matrix((data, (rows, cols)), (count, count)).tocsr().tocoo()
    return matrix.row, matrix.col, matrix.data
