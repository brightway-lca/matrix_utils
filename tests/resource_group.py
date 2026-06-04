from pathlib import Path

import numpy as np
from bw_processing import INDICES_DTYPE, create_datapackage, load_datapackage
from fsspec.implementations.zip import ZipFileSystem

from matrix_utils import ArrayMapper, ResourceGroup
from matrix_utils.indexers import SequentialIndexer
from matrix_utils.utils import safe_concatenate_indices

dirpath = Path(__file__).parent.resolve() / "fixtures"


def get_vector_group():
    return (
        load_datapackage(ZipFileSystem(dirpath / "a-first.zip")).filter_by_attribute(
            "group", "x-third"
        ),
        "x-third",
    )


def get_vector_interface_group():
    return (
        load_datapackage(ZipFileSystem(dirpath / "a-first.zip")).filter_by_attribute(
            "group", "w-fourth"
        ),
        "w-fourth",
    )


def get_array_group():
    return (
        load_datapackage(ZipFileSystem(dirpath / "a-first.zip")).filter_by_attribute(
            "group", "y-second"
        ),
        "y-second",
    )


def get_vector_array_group():
    return (
        load_datapackage(ZipFileSystem(dirpath / "a-first.zip")).filter_by_attribute(
            "group", "z-first"
        ),
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
    class VectorInterface:
        def __next__(self):
            return np.array([10, 20, 30], dtype=float)

    dp, label = get_vector_interface_group()
    dp.rehydrate_interface(label, VectorInterface())
    g = ResourceGroup(package=dp, group_label=label)
    complete_group(g)
    g.calculate()
    assert np.allclose(np.array([10, 20, 30]), g.data_current)
    assert g.data_original.__class__.__name__ == "VectorInterface"


def test_data_current_array_interface():
    class ArrayInterface:
        @property
        def shape(self):
            return (3, 4)

        def __getitem__(self, args):
            return np.array([1, 2, 3], dtype=float) + args[1]

    dp, label = get_vector_array_group()
    dp.rehydrate_interface(label, ArrayInterface())
    g = ResourceGroup(package=dp, group_label=label)
    complete_group(g)
    g.calculate()
    assert np.allclose(np.array([1, 2, 3]), g.data_current)

    next(g.indexer)
    g.calculate()
    assert np.allclose(np.array([2, 3, 4]), g.data_current)
    assert g.data_original.__class__.__name__ == "ArrayInterface"


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


def test_row_col_indices_for_mapping_are_contiguous():
    dp, label = get_array_group()
    g = ResourceGroup(package=dp, group_label=label)

    assert g.row_indices_for_mapping().flags["C_CONTIGUOUS"]
    assert g.col_indices_for_mapping().flags["C_CONTIGUOUS"]
    assert np.array_equal(g.unique_row_indices_for_mapping(), np.array([10, 20, 30]))
    assert np.array_equal(g.unique_col_indices_for_mapping(), np.array([14, 15, 16]))


def make_scaled_group(rescale_array=None, flip_array=None):
    dp = create_datapackage()
    kwargs = dict(
        matrix="foo",
        name="s",
        indices_array=np.array([(0, 1), (2, 3), (4, 5)], dtype=INDICES_DTYPE),
        data_array=np.array([10.0, 20.0, 30.0]),
    )
    if rescale_array is not None:
        kwargs["rescale_array"] = rescale_array
    if flip_array is not None:
        kwargs["flip_array"] = flip_array
    dp.add_persistent_vector(**kwargs)
    g = ResourceGroup(package=dp.filter_by_attribute("group", "s"), group_label="s")
    complete_group(g)
    return g


def test_rescale_applied_static():
    rescale = np.array([0.5, 1.0, 2.0])
    g = make_scaled_group(rescale_array=rescale)
    g.calculate()
    assert np.allclose(g.data_current, [5.0, 20.0, 60.0])


def test_rescale_applied_after_flip():
    rescale = np.array([0.5, 1.0, 2.0])
    flip = np.array([True, False, False])
    g = make_scaled_group(rescale_array=rescale, flip_array=flip)
    g.calculate()
    # flip negates first element, then rescale is applied
    assert np.allclose(g.data_current, [-5.0, 20.0, 60.0])


def test_no_rescale_unchanged():
    g = make_scaled_group()
    g.calculate()
    assert np.allclose(g.data_current, [10.0, 20.0, 30.0])


def test_rescale_applied_array():
    dp = create_datapackage(sequential=True)
    # 3 exchanges, 4 columns (Monte Carlo samples)
    data_array = np.array(
        [[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0], [30.0, 31.0, 32.0, 33.0]]
    )
    rescale_array = np.array([0.5, 1.0, 2.0])
    dp.add_persistent_array(
        matrix="foo",
        name="s",
        indices_array=np.array([(0, 1), (2, 3), (4, 5)], dtype=INDICES_DTYPE),
        data_array=data_array,
        rescale_array=rescale_array,
    )
    g = ResourceGroup(package=dp.filter_by_attribute("group", "s"), group_label="s")
    complete_group(g)

    # column 0: [10, 20, 30] * [0.5, 1.0, 2.0] = [5, 20, 60]
    g.calculate()
    assert np.allclose(g.data_current, [5.0, 20.0, 60.0])

    # column 1: [11, 21, 31] * [0.5, 1.0, 2.0] = [5.5, 21, 62]
    next(g.indexer)
    g.calculate()
    assert np.allclose(g.data_current, [5.5, 21.0, 62.0])


def test_rescale_applied_vector_override():
    rescale = np.array([0.5, 1.0, 2.0])
    g = make_scaled_group(rescale_array=rescale)
    g.calculate(vector=np.array([4.0, 5.0, 6.0]))
    assert np.allclose(g.data_current, [2.0, 5.0, 12.0])


def test_rescale_applied_masked():
    rescale = np.array([0.5, 1.0, 2.0])
    dp = create_datapackage()
    dp.add_persistent_vector(
        matrix="foo",
        name="s",
        indices_array=np.array([(0, 1), (2, 3), (4, 5)], dtype=INDICES_DTYPE),
        data_array=np.array([10.0, 20.0, 30.0]),
        rescale_array=rescale,
    )
    g = ResourceGroup(package=dp.filter_by_attribute("group", "s"), group_label="s")
    g.add_indexer(SequentialIndexer())
    # Mappers that only cover the first two index pairs; (4, 5) will be masked out
    g.add_mapper(0, ArrayMapper(array=np.array([0, 2]), empty_ok=True))
    g.add_mapper(1, ArrayMapper(array=np.array([1, 3]), empty_ok=True))
    g.map_indices()
    g.calculate()
    # Third element masked; [10, 20] * [0.5, 1.0] = [5, 20]
    assert np.allclose(g.data_current, [5.0, 20.0])
