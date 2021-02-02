import pytest
import bw_processing as bwp
import numpy as np


@pytest.fixture
def basic_mm():
    dp = bwp.create_datapackage()
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
        data_array=np.array([1, 2.3, 4, 25]),
    )
    return dp
