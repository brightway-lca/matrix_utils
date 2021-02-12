class MatrixUtilsError(Exception):
    """Base class for bw2calc errors"""

    pass


class AllArraysEmpty(MatrixUtilsError):
    """Can't load the numpy arrays if all of them are empty"""

    pass


class NoArrays(MatrixUtilsError):
    """No arrays for given matrix"""

    pass


class EmptyArray(MatrixUtilsError):
    """Empty array can't be used"""

    pass
