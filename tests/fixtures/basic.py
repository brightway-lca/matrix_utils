import bw_processing as bwp
import numpy as np


def switch_dtype(dp: bwp.Datapackage, indices_32bit: bool) -> tuple:
    if indices_32bit:
        del dp.metadata["64_bit_indices"]
        return [("row", np.int32), ("col", np.int32)]
    else:
        return bwp.INDICES_DTYPE


def basic_mm(indices_32bit: bool = False, **kwargs):
    dp = bwp.create_datapackage(**kwargs)
    dtype = switch_dtype(dp, indices_32bit)
    dp.add_persistent_vector(
        matrix="foo",
        name="vector",
        indices_array=np.array([(0, 0), (2, 1), (4, 2), (8, 3)], dtype=dtype),
        data_array=np.array([1, 2.3, 4, 25]),
    )
    dp.add_persistent_vector(
        matrix="foo",
        name="vector2",
        indices_array=np.array([(10, 10), (12, 9), (14, 8), (18, 7)], dtype=dtype),
        data_array=np.array([11, 12.3, 14, 125]),
    )
    dp.add_persistent_array(
        matrix="foo",
        name="array",
        indices_array=np.array([(1, 0), (2, 1), (5, 1), (8, 1)], dtype=dtype),
        data_array=np.array([[1, 2.3, 4, 25]]).T,
    )
    return dp


def overlapping(indices_32bit: bool = False, **kwargs):
    dp = bwp.create_datapackage(**kwargs)
    dtype = switch_dtype(dp, indices_32bit)
    dp.add_persistent_vector(
        matrix="foo",
        name="vector",
        indices_array=np.array([(0, 0), (2, 1), (4, 2), (8, 3)], dtype=dtype),
        data_array=np.array([1, 2.3, 4, 25]),
    )
    dp.add_persistent_vector(
        matrix="foo",
        name="vector2",
        indices_array=np.array([(0, 0), (12, 9), (2, 1), (18, 7)], dtype=dtype),
        data_array=np.array([11, 12.3, 14, 125]),
    )
    return dp


def aggregation(indices_32bit: bool = False, **kwargs):
    dp = bwp.create_datapackage(**kwargs)
    dtype = switch_dtype(dp, indices_32bit)
    dp.add_persistent_vector(
        matrix="foo",
        name="vector",
        indices_array=np.array([(0, 0), (2, 1), (4, 2), (4, 2), (8, 3)], dtype=dtype),
        data_array=np.array([1, 2.3, 4, 17, 25]),
    )
    return dp


def diagonal(indices_32bit: bool = False, **kwargs):
    dp = bwp.create_datapackage(**kwargs)
    dtype = switch_dtype(dp, indices_32bit)
    dp.add_persistent_vector(
        matrix="foo",
        name="vector",
        indices_array=np.array([(0, 1), (1, 1), (2, 0), (3, 1)], dtype=dtype),
        data_array=np.array([1, 2.3, 4, 25]),
        flip_array=np.array([0, 1, 0, 0], dtype=bool),
    )
    return dp
