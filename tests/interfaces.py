from matrix_utils.errors import EmptyInterface
from matrix_utils import MappedMatrix
from bw_processing import load_datapackage
import pytest
from pathlib import Path
from fs.zipfs import ZipFS


dirpath = Path(__file__).parent.resolve() / "fixtures"


def test_interfaces_not_dehydrated():
    dp = load_datapackage(ZipFS(dirpath / "b-second.zip"))
    with pytest.raises(EmptyInterface):
        MappedMatrix(packages=[dp], matrix="matrix-a")
