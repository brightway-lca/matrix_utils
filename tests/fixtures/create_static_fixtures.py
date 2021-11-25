from pathlib import Path

import numpy as np
from bw_processing import INDICES_DTYPE, UNCERTAINTY_DTYPE, create_datapackage
from fs.zipfs import ZipFS

"""Create fixture to test cross-platform consistency in ordering resource groups.

To do this, we:

* Mix types (static, dynamic; vector, array)
* Provide the same group label in multiple packages
* Intersperse resources for multiple matrices
* Make sure group labels are not alphabetic

Therefore, we will the following resource groups:

z-first:
    dynamic
    array
    matrix-b

y-second:
    static
    array
    matrix-a

x-third:
    static
    vector (with distributions)
    matrix-b

w-fourth
    dynamic
    vector
    matrix-a

This file creates two datapackages. The values in `b-second` will be the same as in  `a-first`, but incremented by 100.
"""


dirpath = Path(__file__).parent.resolve()


class Dummy:
    pass


def add_data(dp, increment=0):
    indices_array = np.array([(10, 14), (20, 15), (30, 16)], dtype=INDICES_DTYPE)
    dp.add_dynamic_array(
        interface=Dummy(),
        matrix="matrix-b",
        name="z-first",
        indices_array=indices_array,
    )

    data_array = np.array([[21, 22, 23], [41, 42, 43], [0, 1, 2]])
    flip_array = np.array([1, 0, 0], dtype=bool)
    dp.add_persistent_array(
        matrix="matrix-a",
        data_array=data_array + increment,
        name="y-second",
        indices_array=indices_array,
        flip_array=flip_array,
    )

    data_array = np.array([101, 111, 112])
    flip_array = np.array([0, 1, 0], dtype=bool)
    indices_array = np.array([(1, 4), (2, 5), (3, 6)], dtype=INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="matrix-b",
        data_array=data_array + increment,
        name="x-third",
        indices_array=indices_array,
        flip_array=flip_array,
    )
    dp.add_dynamic_vector(
        interface=Dummy(),
        indices_array=indices_array,
        matrix="matrix-a",
        name="w-fourth",
    )


def create_ordering_datapackages():
    dp = create_datapackage(
        fs=ZipFS(str(dirpath / "a-first.zip"), write=True),
        name="test-fixture-a",
        id_="fixture-a",
    )
    add_data(dp)
    dp.finalize_serialization()

    dp = create_datapackage(
        fs=ZipFS(str(dirpath / "b-second.zip"), write=True),
        name="test-fixture-b",
        id_="fixture-b",
    )
    add_data(dp)
    dp.finalize_serialization()


def create_sensitivity_fixtures():
    dp = create_datapackage(
        fs=ZipFS(str(dirpath / "sa-1.zip"), write=True),
        name="sa-1",
        id_="sa-1",
        seed=42,
    )

    # x 0 0
    # 0 0 x
    # 0 x 0
    # where x is the column chosen

    indices_array = np.array([(10, 10), (11, 12), (12, 11)], dtype=INDICES_DTYPE)
    dp.add_dynamic_array(
        interface=Dummy(),
        matrix="matrix",
        name="a",
        indices_array=indices_array,
    )

    # -x  0     0
    # 0  x + 1 0
    # 0  0     0
    # where x is column plus 7

    indices_array = np.array([(10, 10), (11, 11)], dtype=INDICES_DTYPE)
    data_array = np.array([[7, 8], [8, 9], [9, 10], [10, 11]]).T
    flip_array = np.array([1, 0], dtype=bool)
    dp.add_persistent_array(
        matrix="matrix",
        data_array=data_array,
        name="b",
        indices_array=indices_array,
        flip_array=flip_array,
    )

    dp.finalize_serialization()

    dp = create_datapackage(
        fs=ZipFS(str(dirpath / "sa-2.zip"), write=True),
        name="sa-2",
        id_="sa-2",
        seed=42,
    )

    # 0 0    0
    # 0 0    0
    # 1 2-1 -3

    data_array = np.array([1, 2, 1, 3])
    flip_array = np.array([0, 0, 1, 1], dtype=bool)
    indices_array = np.array(
        [(12, 10), (12, 11), (12, 11), (12, 12)], dtype=INDICES_DTYPE
    )
    distributions_array = np.zeros((4,), dtype=UNCERTAINTY_DTYPE)
    distributions_array["uncertainty_type"] = (4, 4, 0, 4)
    distributions_array["scale"] = np.NaN
    distributions_array["shape"] = np.NaN
    distributions_array["minimum"] = (0.5, 1.5, np.NaN, 2.5)
    distributions_array["maximum"] = (1.5, 2.5, np.NaN, 3.5)
    distributions_array["loc"] = (1, 2, 1, 3)
    distributions_array["negative"] = (False, False, True, True)
    dp.add_persistent_vector(
        matrix="matrix",
        data_array=data_array,
        name="c",
        indices_array=indices_array,
        flip_array=flip_array,
        distributions_array=distributions_array,
    )

    # 0 0 1
    # 0 0 0
    # 2 0 3

    indices_array = np.array([(10, 12), (12, 10), (12, 12)], dtype=INDICES_DTYPE)
    dp.add_dynamic_vector(
        interface=Dummy(),
        indices_array=indices_array,
        matrix="matrix",
        name="d",
    )

    dp.finalize_serialization()


if __name__ == "__main__":
    create_ordering_datapackages()
    create_sensitivity_fixtures()
