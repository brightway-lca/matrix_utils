from pathlib import Path

import numpy as np
from bw_processing import INDICES_DTYPE, create_datapackage, load_datapackage
from fs.zipfs import ZipFS

from matrix_utils import ArrayMapper, ResourceGroup
from matrix_utils.indexers import SequentialIndexer
from matrix_utils.utils import safe_concatenate_indices

dirpath = Path(__file__).parent.resolve() / "fixtures"


def get_vector_group():
    return (
        load_datapackage(ZipFS(dirpath / "a-first.zip")).filter_by_attribute("group", "x-third"),
        "x-third",
    )


def get_vector_interface_group():
    return (
        load_datapackage(ZipFS(dirpath / "a-first.zip")).filter_by_attribute("group", "w-fourth"),
        "w-fourth",
    )


def get_array_group():
    return (
        load_datapackage(ZipFS(dirpath / "a-first.zip")).filter_by_attribute("group", "y-second"),
        "y-second",
    )


def get_vector_array_group():
    return (
        load_datapackage(ZipFS(dirpath / "a-first.zip")).filter_by_attribute("group", "z-first"),
        "z-first",
    )


def get_empty_group():
    dp = create_datapackage()
    dp.add_persistent_vector(
        matrix="matrix-b",
        data_array=np.array([]),
        name="b",
        indices_array=np.array([], dtype=INDICES_DTYPE),
    )
    return dp, "b"


def complete_group(group):
    group.add_indexer(SequentialIndexer())

    col_mapper = ArrayMapper(
        array=safe_concatenate_indices([group.unique_col_indices_for_mapping()], True),
        empty_ok=True,
    )
    row_mapper = ArrayMapper(
        array=safe_concatenate_indices([group.unique_row_indices_for_mapping()], True),
        empty_ok=True,
    )

    group.add_mapper(0, row_mapper)
    group.add_mapper(1, col_mapper)
    group.map_indices()


def test_data_X_empty():
    dp, label = get_empty_group()
    g = ResourceGroup(package=dp, group_label=label)
    complete_group(g)
    g.calculate()
    assert np.allclose(np.array([]), g.data_current)
    assert np.allclose(np.array([]), g.data_original)


def test_data_current_vector_interface():
    pass


def test_data_current_array_interface():
    pass


def test_data_current_vector():
    dp, label = get_vector_group()
    g = ResourceGroup(package=dp, group_label=label)
    complete_group(g)
    g.calculate()
    assert np.allclose(np.array([101, -111, 112]), g.data_current)
    assert np.allclose(np.array([101, 111, 112]), g.data_original)


def test_data_current_array():
    dp, label = get_array_group()
    g = ResourceGroup(package=dp, group_label=label)
    complete_group(g)
    g.calculate()
    assert np.allclose(np.array([-21, 41, 0]), g.data_current)
    assert np.allclose(np.array([[21, 22, 23], [41, 42, 43], [0, 1, 2]]), g.data_original)
