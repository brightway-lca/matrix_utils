from fixtures import basic_mm
from matrix_utils import MappedMatrix
import bw_processing as bwp
import numpy as np


def test_mappers():
    pass


def test_group_filtering(basic_mm):
    mm = MappedMatrix(
        packages=[basic_mm],
        matrix="foo",
        use_arrays=False,
        use_distributions=False,
    )

    # dp.add_persistent_vector(
    #     matrix="foo",
    #     indices_array=np.array([0, 2, 4, 8]),
    #     data_array = np.array([1, 2.3, 4, 25])
    # )
    # return dp
