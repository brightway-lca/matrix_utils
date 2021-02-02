import pytest
import bw_processing as bwp
import numpy as np


@pytest.fixture
def basic_mm():
    dp = bwp.create_datapackage()
    dp.add_persistent_vector(
        matrix="foo",
        indices_array=np.array([[0, 2, 4, 8], [0, 1, 2, 3]], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([1, 2.3, 4, 25]),
    )
    dp.add_persistent_array(
        matrix="foo",
        indices_array=np.array([[1, 2, 5, 8], [0, 1, 1, 1]], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([1, 2.3, 4, 25]),
    )
    return dp
