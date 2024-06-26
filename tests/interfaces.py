from pathlib import Path

import pytest
from bw_processing import load_datapackage
from fsspec.implementations.zip import ZipFileSystem

from matrix_utils import MappedMatrix
from matrix_utils.errors import EmptyInterface

dirpath = Path(__file__).parent.resolve() / "fixtures"


def test_interfaces_not_dehydrated():
    dp = load_datapackage(ZipFileSystem(dirpath / "b-second.zip"))
    with pytest.raises(EmptyInterface):
        MappedMatrix(packages=[dp], matrix="matrix-a")
