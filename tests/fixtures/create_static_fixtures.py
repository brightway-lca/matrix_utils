from pathlib import Path

import numpy as np
import pandas as pd
from fs.osfs import OSFS
from fs.zipfs import ZipFS

from bw_processing import INDICES_DTYPE, create_datapackage, UNCERTAINTY_DTYPE


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
    # Create the test fixtures

    dirpath = Path(__file__).parent.resolve()

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


if __name__ == "__main__":
    create_ordering_datapackages()
