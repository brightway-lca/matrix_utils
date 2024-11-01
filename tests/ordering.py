from pathlib import Path

import numpy as np
from bw_processing import load_datapackage
from fsspec.implementations.zip import ZipFileSystem

from matrix_utils import MappedMatrix

dirpath = Path(__file__).parent.resolve() / "fixtures"


class Interface:
    def __next__(self):
        return np.arange(3)


def test_ordering():
    dps = [
        load_datapackage(ZipFileSystem(dirpath / "b-second.zip")),
        load_datapackage(ZipFileSystem(dirpath / "a-first.zip")),
    ]
    for dp in dps:
        dp.rehydrate_interface("w-fourth", Interface())

    mm = MappedMatrix(packages=dps, matrix="matrix-a")
    assert [grp.label for grp in mm.groups] == [
        "y-second",
        "w-fourth",
        "y-second",
        "w-fourth",
    ]
