import bw_processing as bwp
import numpy as np


def basic_mm(**kwargs):
    dp = bwp.create_datapackage(**kwargs)
    dp.add_persistent_vector(
        matrix="foo",
        name="vector",
        indices_array=np.array(
            [(0, 0), (2, 1), (4, 2), (8, 3)], dtype=bwp.INDICES_DTYPE
        ),
        data_array=np.array([1, 2.3, 4, 25]),
    )
    dp.add_persistent_vector(
        matrix="foo",
        name="vector2",
        indices_array=np.array(
            [(10, 10), (12, 9), (14, 8), (18, 7)], dtype=bwp.INDICES_DTYPE
        ),
        data_array=np.array([11, 12.3, 14, 125]),
    )
    dp.add_persistent_array(
        matrix="foo",
        name="array",
        indices_array=np.array(
            [(1, 0), (2, 1), (5, 1), (8, 1)], dtype=bwp.INDICES_DTYPE
        ),
        data_array=np.array([[1, 2.3, 4, 25]]).T,
    )
    return dp


def overlapping(**kwargs):
    dp = bwp.create_datapackage(**kwargs)
    dp.add_persistent_vector(
        matrix="foo",
        name="vector",
        indices_array=np.array(
            [(0, 0), (2, 1), (4, 2), (8, 3)], dtype=bwp.INDICES_DTYPE
        ),
        data_array=np.array([1, 2.3, 4, 25]),
    )
    dp.add_persistent_vector(
        matrix="foo",
        name="vector2",
        indices_array=np.array(
            [(0, 0), (12, 9), (2, 1), (18, 7)], dtype=bwp.INDICES_DTYPE
        ),
        data_array=np.array([11, 12.3, 14, 125]),
    )
    return dp


def aggregation(**kwargs):
    dp = bwp.create_datapackage(**kwargs)
    dp.add_persistent_vector(
        matrix="foo",
        name="vector",
        indices_array=np.array(
            [(0, 0), (2, 1), (4, 2), (4, 2), (8, 3)], dtype=bwp.INDICES_DTYPE
        ),
        data_array=np.array([1, 2.3, 4, 17, 25]),
    )
    return dp


def diagonal(**kwargs):
    dp = bwp.create_datapackage(**kwargs)
    dp.add_persistent_vector(
        matrix="foo",
        name="vector",
        indices_array=np.array(
            [(0, 1), (1, 1), (2, 0), (3, 1)], dtype=bwp.INDICES_DTYPE
        ),
        data_array=np.array([1, 2.3, 4, 25]),
        flip_array=np.array([0, 1, 0, 0], dtype=bool),
    )
    return dp
