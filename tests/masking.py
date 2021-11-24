from pathlib import Path

import numpy as np
from bw_processing import INDICES_DTYPE, create_datapackage, load_datapackage
from fs.zipfs import ZipFS

from matrix_utils import ArrayMapper, MappedMatrix

dirpath = Path(__file__).parent.resolve() / "fixtures"


def test_data_original():
    dp = load_datapackage(ZipFS(dirpath / "a-first.zip")).filter_by_attribute(
        "group", "x-third"
    )
    mm = MappedMatrix(packages=[dp], matrix="matrix-b")

    expected = np.array([101, 111, 112])
    for group in mm.groups:
        assert np.allclose(group.data_original, expected)

    mm = MappedMatrix(
        packages=[dp], matrix="matrix-b", custom_filter=lambda x: x["row"] < 3
    )
    for group in mm.groups:
        assert np.allclose(group.data_original, expected)

    am = ArrayMapper(array=np.array([2, 3]))
    mm = MappedMatrix(packages=[dp], matrix="matrix-b", row_mapper=am)
    for group in mm.groups:
        assert np.allclose(group.data_original, expected)


def test_data_current():
    dp = load_datapackage(ZipFS(dirpath / "a-first.zip")).filter_by_attribute(
        "group", "x-third"
    )
    expected = np.array([101, -111, 112])
    mm = MappedMatrix(packages=[dp], matrix="matrix-b")
    for group in mm.groups:
        assert np.allclose(group.data_current, expected)

    dp = load_datapackage(ZipFS(dirpath / "a-first.zip")).filter_by_attribute(
        "group", "x-third"
    )
    expected = np.array([101, -111])
    mm = MappedMatrix(
        packages=[dp], matrix="matrix-b", custom_filter=lambda x: x["row"] < 3
    )
    for group in mm.groups:
        assert np.allclose(group.data_current, expected)

    dp = load_datapackage(ZipFS(dirpath / "a-first.zip")).filter_by_attribute(
        "group", "x-third"
    )
    expected = np.array([-111, 112])
    am = ArrayMapper(array=np.array([2, 3]))
    mm = MappedMatrix(packages=[dp], matrix="matrix-b", row_mapper=am)
    for group in mm.groups:
        assert np.allclose(group.data_current, expected)

    dp = load_datapackage(ZipFS(dirpath / "a-first.zip")).filter_by_attribute(
        "group", "x-third"
    )
    expected = np.array([-111])
    am = ArrayMapper(array=np.array([2, 3]))
    mm = MappedMatrix(
        packages=[dp],
        matrix="matrix-b",
        row_mapper=am,
        custom_filter=lambda x: x["row"] < 3,
    )
    for group in mm.groups:
        assert np.allclose(group.data_current, expected)


def test_indices_mapped():
    dp = load_datapackage(ZipFS(dirpath / "a-first.zip")).filter_by_attribute(
        "group", "x-third"
    )

    expected = np.array([0, 1, 2])
    mm = MappedMatrix(packages=[dp], matrix="matrix-b")
    for group in mm.groups:
        assert np.allclose(group.row_mapped, expected)
        assert np.allclose(group.col_mapped, expected)

    expected = np.array([0, 1, -1])
    mm = MappedMatrix(
        packages=[dp], matrix="matrix-b", custom_filter=lambda x: x["row"] < 3
    )
    for group in mm.groups:
        assert np.allclose(group.row_mapped, expected)
        assert np.allclose(group.col_mapped, expected)

    rexpected = np.array([-1, 0, 1])
    cexpected = np.array([0, 1, 2])
    am = ArrayMapper(array=np.array([2, 3]))
    mm = MappedMatrix(packages=[dp], matrix="matrix-b", row_mapper=am)
    for group in mm.groups:
        assert np.allclose(group.row_mapped, rexpected)
        assert np.allclose(group.col_mapped, cexpected)

    rexpected = np.array([-1, 0, 1])
    cexpected = np.array([0, 1, -1])
    mm = MappedMatrix(
        packages=[dp],
        matrix="matrix-b",
        row_mapper=am,
        custom_filter=lambda x: x["row"] < 3,
    )
    for group in mm.groups:
        assert np.allclose(group.row_mapped, rexpected)
        assert np.allclose(group.col_mapped, cexpected)


def test_indices_masked():
    dp = load_datapackage(ZipFS(dirpath / "a-first.zip")).filter_by_attribute(
        "group", "x-third"
    )

    expected = np.array([0, 1, 2])
    mm = MappedMatrix(packages=[dp], matrix="matrix-b")
    for group in mm.groups:
        assert np.allclose(group.row_masked, expected)
        assert np.allclose(group.col_masked, expected)

    expected = np.array([0, 1])
    mm = MappedMatrix(
        packages=[dp], matrix="matrix-b", custom_filter=lambda x: x["row"] < 3
    )
    for group in mm.groups:
        assert np.allclose(group.row_masked, expected)
        assert np.allclose(group.col_masked, expected)

    rexpected = np.array([0, 1])
    cexpected = np.array([1, 2])
    am = ArrayMapper(array=np.array([2, 3]))
    mm = MappedMatrix(packages=[dp], matrix="matrix-b", row_mapper=am)
    for group in mm.groups:
        assert np.allclose(group.row_masked, rexpected)
        assert np.allclose(group.col_masked, cexpected)

    rexpected = np.array([0])
    cexpected = np.array([1])
    mm = MappedMatrix(
        packages=[dp],
        matrix="matrix-b",
        row_mapper=am,
        custom_filter=lambda x: x["row"] < 3,
    )
    for group in mm.groups:
        assert np.allclose(group.row_masked, rexpected)
        assert np.allclose(group.col_masked, cexpected)


def test_indices_matrix_without_aggregation():
    dp = create_datapackage(sum_intra_duplicates=False)
    data_array = np.array([101, 111, 112, 113])
    indices_array = np.array([(1, 4), (2, 5), (3, 6), (3, 6)], dtype=INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="matrix-b",
        data_array=data_array,
        name="x-third",
        indices_array=indices_array,
    )

    expected = np.array([0, 1, 2, 2])
    mm = MappedMatrix(packages=[dp], matrix="matrix-b")
    for group in mm.groups:
        assert np.allclose(group.row_matrix, expected)
        assert np.allclose(group.col_matrix, expected)

    expected = np.array([0, 1])
    mm = MappedMatrix(
        packages=[dp], matrix="matrix-b", custom_filter=lambda x: x["row"] < 3
    )
    for group in mm.groups:
        assert np.allclose(group.row_matrix, expected)
        assert np.allclose(group.col_matrix, expected)

    rexpected = np.array([0, 1, 1])
    cexpected = np.array([1, 2, 2])
    am = ArrayMapper(array=np.array([2, 3]))
    mm = MappedMatrix(packages=[dp], matrix="matrix-b", row_mapper=am)
    for group in mm.groups:
        assert np.allclose(group.row_matrix, rexpected)
        assert np.allclose(group.col_matrix, cexpected)

    rexpected = np.array([0])
    cexpected = np.array([1])
    mm = MappedMatrix(
        packages=[dp],
        matrix="matrix-b",
        row_mapper=am,
        custom_filter=lambda x: x["row"] < 3,
    )
    for group in mm.groups:
        assert np.allclose(group.row_matrix, rexpected)
        assert np.allclose(group.col_matrix, cexpected)


def test_indices_matrix_with_aggregation():
    dp = create_datapackage(sum_intra_duplicates=True)
    data_array = np.array([101, 111, 112, 113])
    indices_array = np.array([(1, 4), (2, 5), (3, 6), (3, 6)], dtype=INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="matrix-b",
        data_array=data_array,
        name="x-third",
        indices_array=indices_array,
    )

    expected = np.array([0, 1, 2])
    mm = MappedMatrix(packages=[dp], matrix="matrix-b")
    for group in mm.groups:
        assert np.allclose(group.row_matrix, expected)
        assert np.allclose(group.col_matrix, expected)

    expected = np.array([0, 1])
    mm = MappedMatrix(
        packages=[dp], matrix="matrix-b", custom_filter=lambda x: x["row"] < 3
    )
    for group in mm.groups:
        assert np.allclose(group.row_matrix, expected)
        assert np.allclose(group.col_matrix, expected)

    rexpected = np.array([0, 1])
    cexpected = np.array([1, 2])
    am = ArrayMapper(array=np.array([2, 3]))
    mm = MappedMatrix(packages=[dp], matrix="matrix-b", row_mapper=am)
    for group in mm.groups:
        assert np.allclose(group.row_matrix, rexpected)
        assert np.allclose(group.col_matrix, cexpected)

    rexpected = np.array([0])
    cexpected = np.array([1])
    mm = MappedMatrix(
        packages=[dp],
        matrix="matrix-b",
        row_mapper=am,
        custom_filter=lambda x: x["row"] < 3,
    )
    for group in mm.groups:
        assert np.allclose(group.row_matrix, rexpected)
        assert np.allclose(group.col_matrix, cexpected)


def test_indices_transposed():
    dp = load_datapackage(ZipFS(dirpath / "a-first.zip")).filter_by_attribute(
        "group", "x-third"
    )

    cexpected = np.array([-1, 0, 1])
    rexpected = np.array([0, 1, -1])
    am = ArrayMapper(array=np.array([2, 3]))
    mm = MappedMatrix(
        packages=[dp],
        matrix="matrix-b",
        col_mapper=am,
        custom_filter=lambda x: x["col"] < 3,
        transpose=True,
    )
    for group in mm.groups:
        assert np.allclose(group.row_mapped, rexpected)
        assert np.allclose(group.col_mapped, cexpected)


def test_flip_masked():
    dp = load_datapackage(ZipFS(dirpath / "a-first.zip")).filter_by_attribute(
        "group", "x-third"
    )

    expected = np.array([True])
    am = ArrayMapper(array=np.array([2, 3]))
    mm = MappedMatrix(
        packages=[dp],
        matrix="matrix-b",
        row_mapper=am,
        custom_filter=lambda x: x["row"] < 3,
    )
    for group in mm.groups:
        assert np.allclose(group.flip, expected)
