from pathlib import Path

import bw_processing as bwp
import numpy as np
import pytest
from fixtures import basic_mm, diagonal
from fsspec.implementations.zip import ZipFileSystem

from matrix_utils import ArrayMapper, MappedMatrix
from matrix_utils.errors import AllArraysEmpty, EmptyArray

# --- input_params ---


def _mm_with_params(**kwargs):
    """Build a MappedMatrix from the given datapackages."""
    return MappedMatrix(
        matrix="foo",
        use_vectors=True,
        use_arrays=True,
        use_distributions=False,
        **kwargs,
    )


def test_input_params_empty_when_no_params():
    mm = _mm_with_params(packages=[basic_mm()])
    assert mm.input_params() == {}


def test_input_params_vector():
    dp = bwp.create_datapackage()
    params = np.array([0.1, 0.9])
    dp.add_persistent_vector(
        matrix="foo",
        name="v",
        indices_array=np.array([(0, 0), (1, 1)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([5.0, 7.0]),
        params_array=params,
    )
    mm = _mm_with_params(packages=[dp])
    result = mm.input_params()
    pkg = list(mm.packages.keys())[0]
    assert list(result.keys()) == [(pkg, "v")]
    assert np.allclose(result[(pkg, "v")], params)


def test_input_params_only_groups_with_params_included():
    dp = bwp.create_datapackage()
    dp.add_persistent_vector(
        matrix="foo",
        name="with_params",
        indices_array=np.array([(0, 0)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([1.0]),
        params_array=np.array([42.0]),
    )
    dp.add_persistent_vector(
        matrix="foo",
        name="no_params",
        indices_array=np.array([(1, 1)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([2.0]),
    )
    mm = _mm_with_params(packages=[dp])
    result = mm.input_params()
    labels = {label for (_, label) in result}
    assert labels == {"with_params"}


def test_input_params_array_tracks_indexer():
    dp = bwp.create_datapackage(sequential=True)
    data = np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])
    params = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dp.add_persistent_array(
        matrix="foo",
        name="a",
        indices_array=np.array([(0, 0), (1, 1)], dtype=bwp.INDICES_DTYPE),
        data_array=data,
        params_array=params,
    )
    mm = _mm_with_params(packages=[dp])
    pkg = list(mm.packages.keys())[0]
    assert np.allclose(mm.input_params()[(pkg, "a")], [1.0, 4.0])
    next(mm)
    assert np.allclose(mm.input_params()[(pkg, "a")], [2.0, 5.0])
    next(mm)
    assert np.allclose(mm.input_params()[(pkg, "a")], [3.0, 6.0])


def test_input_params_duplicate_labels_across_packages():
    dp1 = bwp.create_datapackage()
    dp1.add_persistent_vector(
        matrix="foo",
        name="v",
        indices_array=np.array([(0, 0)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([1.0]),
        params_array=np.array([0.1]),
    )
    dp2 = bwp.create_datapackage()
    dp2.add_persistent_vector(
        matrix="foo",
        name="v",
        indices_array=np.array([(1, 1)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([2.0]),
        params_array=np.array([0.9]),
    )
    mm = _mm_with_params(packages=[dp1, dp2])
    result = mm.input_params()
    assert len(result) == 2
    values = list(result.values())
    assert any(np.allclose(v, [0.1]) for v in values)
    assert any(np.allclose(v, [0.9]) for v in values)


dirpath = Path(__file__).parent.resolve() / "fixtures"


def test_mappers():
    given = np.array([4, 2, 1, 0])
    expected = np.array([2, 1, -1, 0])
    mm = MappedMatrix(
        packages=[basic_mm()],
        matrix="foo",
        use_arrays=False,
        use_distributions=False,
    )
    result = mm.row_mapper.map_array(given)
    assert np.allclose(result, expected)


def test_group_filtering():
    mm = MappedMatrix(
        packages=[basic_mm()],
        matrix="foo",
        use_arrays=False,
        use_distributions=False,
    )
    assert len(mm.groups) == 2
    assert mm.groups[0].label == "vector"
    assert mm.groups[1].label == "vector2"


def test_transpose_matrix():
    mm = MappedMatrix(
        packages=[diagonal()],
        matrix="foo",
        use_arrays=False,
        use_distributions=False,
        transpose=True,
    )
    assert mm.matrix.shape == (2, 4)
    assert np.allclose(mm.matrix.data, [4, 1, -2.3, 25])


def test_diagonal_matrix():
    mm = MappedMatrix(
        packages=[diagonal()],
        matrix="foo",
        use_arrays=False,
        use_distributions=False,
    )
    assert mm.matrix.shape == (4, 2)
    assert np.allclose(mm.matrix.data, [1, -2.3, 4, 25])

    mm = MappedMatrix(
        packages=[diagonal()],
        matrix="foo",
        use_arrays=False,
        use_distributions=False,
        diagonal=True,
    )
    assert mm.matrix.shape == (4, 4)
    for x, y in zip(range(4), [1, -2.3, 4, 25]):
        assert mm.matrix[x, x] == y

    assert np.allclose(mm.matrix.data, [1, -2.3, 4, 25])


def test_custom_filter():
    mm = MappedMatrix(
        packages=[diagonal()],
        matrix="foo",
        use_arrays=False,
        use_distributions=False,
    )
    assert mm.matrix.shape == (4, 2)

    mm = MappedMatrix(
        packages=[diagonal()],
        matrix="foo",
        use_arrays=False,
        use_distributions=False,
        custom_filter=lambda x: x["col"] == 1,
    )
    assert mm.matrix.shape == (3, 1)
    assert mm.matrix.sum() == 1 - 2.3 + 25

    with pytest.raises(EmptyArray):
        mm = MappedMatrix(
            packages=[diagonal()],
            matrix="foo",
            use_arrays=False,
            use_distributions=False,
            custom_filter=lambda x: x["col"] == 2,
        )


def test_indexer_override():
    s = bwp.create_datapackage(sequential=True)
    s.add_persistent_array(
        matrix="foo",
        data_array=np.arange(12).reshape(3, 4),
        indices_array=np.array([(0, 0), (1, 1), (0, 1)], dtype=bwp.INDICES_DTYPE),
    )
    mm = MappedMatrix(
        packages=[s],
        matrix="foo",
        use_arrays=True,
        use_distributions=False,
    )
    assert np.allclose(mm.matrix.toarray(), [[0, 8], [0, 4]])
    next(mm)
    assert np.allclose(mm.matrix.toarray(), [[1, 9], [0, 5]])
    next(mm)
    assert np.allclose(mm.matrix.toarray(), [[2, 10], [0, 6]])

    class MyIndexer:
        index = 2

        def __next__(self):
            pass

    mm = MappedMatrix(
        packages=[s],
        matrix="foo",
        use_arrays=True,
        use_distributions=False,
        indexer_override=MyIndexer(),
    )
    assert np.allclose(mm.matrix.toarray(), [[2, 10], [0, 6]])
    next(mm)
    assert np.allclose(mm.matrix.toarray(), [[2, 10], [0, 6]])
    next(mm)
    assert np.allclose(mm.matrix.toarray(), [[2, 10], [0, 6]])


def test_no_packages_error():
    with pytest.raises(AllArraysEmpty) as exc_info:
        MappedMatrix(
            packages=[],
            matrix="foo",
        )
    assert (
        exc_info.value.args[0]
        == """
No data found to build foo matrix.

No datapackages found which could provide data to build this matrix.
"""
    )


def test_no_packages_empty_ok():
    mm = MappedMatrix(packages=[], matrix="foo", empty_ok=True)
    assert mm.matrix.shape == (0, 0)


def test_no_useful_packages_empty_ok():
    mm = MappedMatrix(packages=[diagonal()], matrix="bar", empty_ok=True)
    assert mm.matrix.shape == (0, 0)


def test_existing_indexer():
    class MyIndexer:
        index = 2

        def __next__(self):
            pass

    s = bwp.create_datapackage(sequential=True)
    s.add_persistent_array(
        matrix="foo",
        data_array=np.arange(12).reshape(3, 4),
        indices_array=np.array([(0, 0), (1, 1), (0, 1)], dtype=bwp.INDICES_DTYPE),
    )
    s.indexer = MyIndexer()
    mm = MappedMatrix(
        packages=[s],
        matrix="foo",
        use_arrays=True,
        use_distributions=False,
    )
    assert np.allclose(mm.matrix.toarray(), [[2, 10], [0, 6]])
    next(mm)
    assert np.allclose(mm.matrix.toarray(), [[2, 10], [0, 6]])


def test_reset_indexers():
    s = bwp.create_datapackage(sequential=True)
    s.add_persistent_array(
        matrix="foo",
        data_array=np.arange(12).reshape(3, 4),
        indices_array=np.array([(0, 0), (1, 1), (0, 1)], dtype=bwp.INDICES_DTYPE),
    )
    mm = MappedMatrix(
        packages=[s],
        matrix="foo",
        use_arrays=True,
        use_distributions=False,
    )
    assert np.allclose(mm.matrix.toarray(), [[0, 8], [0, 4]])
    next(mm)
    assert np.allclose(mm.matrix.toarray(), [[1, 9], [0, 5]])
    next(mm)
    assert np.allclose(mm.matrix.toarray(), [[2, 10], [0, 6]])
    next(mm)
    mm.reset_indexers()
    next(mm)
    assert np.allclose(mm.matrix.toarray(), [[1, 9], [0, 5]])


def test_reset_indexers_rebuild():
    s = bwp.create_datapackage(sequential=True)
    s.add_persistent_array(
        matrix="foo",
        data_array=np.arange(12).reshape(3, 4),
        indices_array=np.array([(0, 0), (1, 1), (0, 1)], dtype=bwp.INDICES_DTYPE),
    )
    mm = MappedMatrix(
        packages=[s],
        matrix="foo",
        use_arrays=True,
        use_distributions=False,
    )
    assert np.allclose(mm.matrix.toarray(), [[0, 8], [0, 4]])
    next(mm)
    assert np.allclose(mm.matrix.toarray(), [[1, 9], [0, 5]])
    next(mm)
    mm.reset_indexers(rebuild=True)
    assert np.allclose(mm.matrix.toarray(), [[0, 8], [0, 4]])


@pytest.fixture
def sensitivity_dps():
    class VectorInterface:
        def __next__(self):
            return np.array([1, 2, 3])

    class ArrayInterface:
        @property
        def shape(self):
            return (3, 100)

        def __getitem__(self, args):
            return np.ones((3,)) * args[1]

    dp_1 = bwp.load_datapackage(ZipFileSystem(dirpath / "sa-1.zip"))
    dp_1.rehydrate_interface("a", ArrayInterface())

    dp_2 = bwp.load_datapackage(ZipFileSystem(dirpath / "sa-2.zip"))
    dp_2.rehydrate_interface("d", VectorInterface())

    return dp_1, dp_2


def test_matrix_building_multiple_dps_no_distributions_no_arrays_replacement(
    sensitivity_dps,
):
    sensitivity_dps[1].metadata["sum_intra_duplicates"] = False
    mm = MappedMatrix(
        packages=sensitivity_dps,
        matrix="matrix",
        use_vectors=True,
        use_arrays=False,
        use_distributions=False,
    )
    expected = np.array(
        [
            [0, 0, 1],
            [2, -1, 3],
        ]
    )
    assert np.allclose(mm.matrix.toarray(), expected)


def test_matrix_building_multiple_dps_no_distributions_no_arrays_replacement_summing_intra(
    sensitivity_dps,
):
    mm = MappedMatrix(
        packages=sensitivity_dps,
        matrix="matrix",
        use_vectors=True,
        use_arrays=False,
        use_distributions=False,
    )
    expected = np.array(
        [
            [0, 0, 1],
            [2, 1, 3],
        ]
    )
    assert np.allclose(mm.matrix.toarray(), expected)


def test_matrix_building_multiple_dps_no_distributions_no_arrays_summing_inter_within_package(
    sensitivity_dps,
):
    sensitivity_dps[1].metadata["sum_inter_duplicates"] = True
    mm = MappedMatrix(
        packages=sensitivity_dps,
        matrix="matrix",
        use_vectors=True,
        use_arrays=False,
        use_distributions=False,
    )
    expected = np.array(
        [
            [0, 0, 1],
            [3, 1, 0],
        ]
    )
    assert np.allclose(mm.matrix.toarray(), expected)


def test_matrix_building_multiple_dps_yes_distributions_no_arrays_replacement(
    sensitivity_dps,
):
    sensitivity_dps[1].metadata["sum_intra_duplicates"] = False
    mm = MappedMatrix(
        packages=sensitivity_dps,
        matrix="matrix",
        use_vectors=True,
        use_arrays=False,
        use_distributions=True,
    )

    expected = np.array(
        [
            [0, 0, 1],
            [2, -1, 3],
        ]
    )
    assert np.allclose(mm.matrix.toarray(), expected)


def test_matrix_building_multiple_dps_yes_distributions_no_arrays_sum_both_within_package(
    sensitivity_dps,
):
    sensitivity_dps[1].metadata["sum_inter_duplicates"] = True
    mm = MappedMatrix(
        packages=sensitivity_dps,
        matrix="matrix",
        use_vectors=True,
        use_arrays=False,
        use_distributions=True,
    )

    sample = [0.87454012, 2.45071431, 1.0, 3.23199394]
    expected = np.array(
        [
            [0, 0, 1],
            [2 + sample[0], sample[1] - sample[2], 3 - sample[3]],
        ]
    )
    assert np.allclose(mm.matrix.toarray(), expected)


def test_matrix_building_multiple_dps_yes_distributions_no_arrays_sum_inter_only(
    sensitivity_dps,
):
    sensitivity_dps[1].metadata["sum_inter_duplicates"] = True
    sensitivity_dps[1].metadata["sum_intra_duplicates"] = False
    mm = MappedMatrix(
        packages=sensitivity_dps,
        matrix="matrix",
        use_vectors=True,
        use_arrays=False,
        use_distributions=True,
    )

    sample = [0.87454012, 2.45071431, 1.0, 3.23199394]
    expected = np.array(
        [
            [0, 0, 1],
            [2 + sample[0], -sample[2], 3 - sample[3]],
        ]
    )
    assert np.allclose(mm.matrix.toarray(), expected)


def test_matrix_building_multiple_dps_yes_distributions_no_arrays_sum_intra_only(
    sensitivity_dps,
):
    mm = MappedMatrix(
        packages=sensitivity_dps,
        matrix="matrix",
        use_vectors=True,
        use_arrays=False,
        use_distributions=True,
    )

    sample = [0.87454012, 2.45071431, 1.0, 3.23199394]
    expected = np.array(
        [
            [0, 0, 1],
            [2, sample[1] - sample[2], 3],
        ]
    )
    assert np.allclose(mm.matrix.toarray(), expected)


def test_matrix_building_multiple_dps_no_distributions_yes_arrays_replacement(
    sensitivity_dps,
):
    indices = [
        191664963,
        1662057957,
        1405681631,
    ]
    sensitivity_dps[0].metadata["sum_intra_duplicates"] = False
    sensitivity_dps[1].metadata["sum_intra_duplicates"] = False
    mm = MappedMatrix(
        packages=sensitivity_dps,
        matrix="matrix",
        use_vectors=True,
        use_arrays=True,
        use_distributions=False,
    )

    for index in indices:
        expected = np.array(
            [
                [-(index % 4 + 7), 0, 1],
                [0, (index % 4 + 8), index % 100],
                [2, -1, 3],
            ]
        )
        assert np.allclose(mm.matrix.toarray(), expected)
        next(mm)


def test_matrix_building_multiple_dps_no_distributions_yes_arrays_sum_intra_only(
    sensitivity_dps,
):
    indices = [
        191664963,
        1662057957,
        1405681631,
    ]
    mm = MappedMatrix(
        packages=sensitivity_dps,
        matrix="matrix",
        use_vectors=True,
        use_arrays=True,
        use_distributions=False,
    )

    for index in indices:
        expected = np.array(
            [
                [-(index % 4 + 7), 0, 1],
                [0, (index % 4 + 8), index % 100],
                [2, 2 - 1, 3],
            ]
        )
        assert np.allclose(mm.matrix.toarray(), expected)
        next(mm)


def test_matrix_building_multiple_dps_no_distributions_yes_arrays_sum_inter_only(
    sensitivity_dps,
):
    indices = [
        191664963,
        1662057957,
        1405681631,
    ]
    sensitivity_dps[0].metadata["sum_intra_duplicates"] = False
    sensitivity_dps[1].metadata["sum_intra_duplicates"] = False
    sensitivity_dps[0].metadata["sum_inter_duplicates"] = True
    sensitivity_dps[1].metadata["sum_inter_duplicates"] = True
    mm = MappedMatrix(
        packages=sensitivity_dps,
        matrix="matrix",
        use_vectors=True,
        use_arrays=True,
        use_distributions=False,
    )

    for index in indices:
        expected = np.array(
            [
                [index % 100 - (index % 4 + 7), 0, 1],
                [0, (index % 4 + 8), index % 100],
                [1 + 2, index % 100 - 1, 3 - 3],
            ]
        )
        assert np.allclose(mm.matrix.toarray(), expected)
        next(mm)


def test_matrix_building_multiple_dps_no_distributions_yes_arrays_sum_both(
    sensitivity_dps,
):
    indices = [
        191664963,
        1662057957,
        1405681631,
    ]
    sensitivity_dps[0].metadata["sum_inter_duplicates"] = True
    sensitivity_dps[1].metadata["sum_inter_duplicates"] = True
    mm = MappedMatrix(
        packages=sensitivity_dps,
        matrix="matrix",
        use_vectors=True,
        use_arrays=True,
        use_distributions=False,
    )

    for index in indices:
        expected = np.array(
            [
                [index % 100 - (index % 4 + 7), 0, 1],
                [0, (index % 4 + 8), index % 100],
                [1 + 2, 2 + index % 100 - 1, 3 - 3],
            ]
        )
        assert np.allclose(mm.matrix.toarray(), expected)
        next(mm)


def test_input_data_vector(sensitivity_dps):
    indices = [
        191664963,
        1662057957,
        1405681631,
    ]
    mm = MappedMatrix(
        packages=sensitivity_dps,
        matrix="matrix",
        use_vectors=True,
        use_arrays=True,
        use_distributions=False,
    )

    for index in indices:
        expected = np.array(
            [
                index % 100,
                index % 100,
                index % 100,
                -(index % 4 + 7),
                (index % 4 + 8),
                1,
                2,
                -1,
                -3,
                1,
                2,
                3,
            ]
        )
        assert np.allclose(mm.input_data_vector(), expected)
        next(mm)


def test_input_row_col_indices(sensitivity_dps):
    mm = MappedMatrix(
        packages=sensitivity_dps,
        matrix="matrix",
        use_vectors=True,
        use_arrays=True,
        use_distributions=False,
    )

    expected = np.array(
        [
            (0, 0),
            (1, 2),
            (2, 1),
            (0, 0),
            (1, 1),
            (2, 0),
            (2, 1),
            (2, 1),
            (2, 2),
            (0, 2),
            (2, 0),
            (2, 2),
        ],
        dtype=bwp.INDICES_DTYPE,
    )
    assert np.allclose(mm.input_row_col_indices()["row"], expected["row"])
    assert np.allclose(mm.input_row_col_indices()["col"], expected["col"])


def test_input_provenance(sensitivity_dps):
    mm = MappedMatrix(
        packages=sensitivity_dps,
        matrix="matrix",
        use_vectors=True,
        use_arrays=True,
        use_distributions=False,
    )

    expected = [
        (sensitivity_dps[0], "a", (0, 3)),
        (sensitivity_dps[0], "b", (3, 5)),
        (sensitivity_dps[1], "c", (5, 9)),
        (sensitivity_dps[1], "d", (9, 12)),
    ]
    assert mm.input_provenance() == expected


def test_input_uncertainties(sensitivity_dps):
    mm = MappedMatrix(
        packages=sensitivity_dps,
        matrix="matrix",
        use_vectors=True,
        use_arrays=True,
        use_distributions=True,
    )

    magic = 191664963

    expected = np.array(
        [
            (99, magic % 100, np.nan, np.nan, np.nan, np.nan, False),
            (99, magic % 100, np.nan, np.nan, np.nan, np.nan, False),
            (99, magic % 100, np.nan, np.nan, np.nan, np.nan, False),
            (98, -8.5, 1.118034, np.nan, np.nan, np.nan, False),
            (98, 9.5, 1.118034, np.nan, np.nan, np.nan, False),
            (4, 1, np.nan, np.nan, 0.5, 1.5, False),
            (4, 2, np.nan, np.nan, 1.5, 2.5, False),
            (0, 1, np.nan, np.nan, np.nan, np.nan, True),
            (4, 3, np.nan, np.nan, 2.5, 3.5, True),
            (99, 1, np.nan, np.nan, np.nan, np.nan, False),
            (99, 2, np.nan, np.nan, np.nan, np.nan, False),
            (99, 3, np.nan, np.nan, np.nan, np.nan, False),
        ],
        dtype=bwp.UNCERTAINTY_DTYPE,
    )
    ua = mm.input_uncertainties()
    for field, _ in bwp.UNCERTAINTY_DTYPE:
        assert np.allclose(ua[field], expected[field], equal_nan=True)


def test_input_uncertainties_limit_samples(sensitivity_dps):
    mm = MappedMatrix(
        packages=sensitivity_dps,
        matrix="matrix",
        use_vectors=True,
        use_arrays=True,
        use_distributions=True,
    )

    magic = 191664963

    expected = np.array(
        [
            (99, magic % 100, np.nan, np.nan, np.nan, np.nan, False),
            (99, magic % 100, np.nan, np.nan, np.nan, np.nan, False),
            (99, magic % 100, np.nan, np.nan, np.nan, np.nan, False),
            (98, -8, 0.8164966, np.nan, np.nan, np.nan, False),
            (98, 9, 0.8164966, np.nan, np.nan, np.nan, False),
            (4, 1, np.nan, np.nan, 0.5, 1.5, False),
            (4, 2, np.nan, np.nan, 1.5, 2.5, False),
            (0, 1, np.nan, np.nan, np.nan, np.nan, True),
            (4, 3, np.nan, np.nan, 2.5, 3.5, True),
            (99, 1, np.nan, np.nan, np.nan, np.nan, False),
            (99, 2, np.nan, np.nan, np.nan, np.nan, False),
            (99, 3, np.nan, np.nan, np.nan, np.nan, False),
        ],
        dtype=bwp.UNCERTAINTY_DTYPE,
    )
    ua = mm.input_uncertainties(number_samples=3)
    for field, _ in bwp.UNCERTAINTY_DTYPE:
        assert np.allclose(ua[field], expected[field], equal_nan=True)


def test_input_uncertainties_no_distributions(sensitivity_dps):
    mm = MappedMatrix(
        packages=sensitivity_dps,
        matrix="matrix",
        use_vectors=True,
        use_arrays=True,
        use_distributions=False,
    )

    magic = 191664963

    expected = np.array(
        [
            (99, magic % 100, np.nan, np.nan, np.nan, np.nan, False),
            (99, magic % 100, np.nan, np.nan, np.nan, np.nan, False),
            (99, magic % 100, np.nan, np.nan, np.nan, np.nan, False),
            (98, -8.5, 1.118034, np.nan, np.nan, np.nan, False),
            (98, 9.5, 1.118034, np.nan, np.nan, np.nan, False),
            (0, 1, np.nan, np.nan, np.nan, np.nan, False),
            (0, 2, np.nan, np.nan, np.nan, np.nan, False),
            (0, -1, np.nan, np.nan, np.nan, np.nan, False),
            (0, -3, np.nan, np.nan, np.nan, np.nan, False),
            (99, 1, np.nan, np.nan, np.nan, np.nan, False),
            (99, 2, np.nan, np.nan, np.nan, np.nan, False),
            (99, 3, np.nan, np.nan, np.nan, np.nan, False),
        ],
        dtype=bwp.UNCERTAINTY_DTYPE,
    )
    ua = mm.input_uncertainties()
    for field, _ in bwp.UNCERTAINTY_DTYPE:
        assert np.allclose(ua[field], expected[field], equal_nan=True)


def test_input_uncertainties_array_group_without_flip():
    """Regression test for issue #8: array group without flip should not crash."""
    dp = bwp.create_datapackage()
    indices_array = np.array([(10, 10), (11, 11)], dtype=bwp.INDICES_DTYPE)
    data_array = np.array([[7, 8], [8, 9], [9, 10]]).T
    dp.add_persistent_array(
        matrix="matrix",
        data_array=data_array,
        name="no_flip",
        indices_array=indices_array,
    )
    mm = MappedMatrix(
        packages=[dp],
        matrix="matrix",
        use_vectors=True,
        use_arrays=True,
        use_distributions=False,
    )
    ua = mm.input_uncertainties()
    assert ua.shape == (2,)
    assert np.allclose(ua["loc"], [8.0, 9.0])


def test_input_index_vector(sensitivity_dps):
    dp = bwp.create_datapackage(combinatorial=True)

    indices_array = np.array([(10, 10), (11, 11)], dtype=bwp.INDICES_DTYPE)
    data_array = np.array([[7, 8], [8, 9]]).T
    flip_array = np.array([1, 0], dtype=bool)
    dp.add_persistent_array(
        matrix="matrix",
        data_array=data_array,
        name="g",
        indices_array=indices_array,
        flip_array=flip_array,
    )
    dp.add_persistent_array(
        matrix="matrix",
        data_array=data_array,
        name="h",
        indices_array=indices_array,
        flip_array=flip_array,
    )

    mm = MappedMatrix(
        packages=list(sensitivity_dps) + [dp],
        matrix="matrix",
        use_vectors=True,
        use_arrays=True,
        use_distributions=False,
    )

    expected = [
        (191664963, 191664963, 0, 0),
        (1662057957, 1662057957, 0, 1),
        (1405681631, 1405681631, 1, 0),
        (942484272, 942484272, 1, 1),
    ]
    for row in expected:
        assert np.allclose(mm.input_indexer_vector(), row)
        try:
            next(mm)
        except StopIteration:
            pass


def test_all_empty_after_custom_filter():
    s = bwp.create_datapackage()
    s.add_persistent_vector(
        matrix="foo",
        data_array=np.arange(2),
        indices_array=np.array([(0, 0), (1, 0)], dtype=bwp.INDICES_DTYPE),
    )
    with pytest.raises(EmptyArray):
        MappedMatrix(packages=[s], matrix="foo", custom_filter=lambda x: x["col"] > 0)

    assert MappedMatrix(
        packages=[s], matrix="foo", custom_filter=lambda x: x["col"] > 0, empty_ok=True
    )


def test_one_empty_after_custom_filter():
    s = bwp.create_datapackage()
    s.add_persistent_vector(
        matrix="foo",
        data_array=np.arange(2),
        indices_array=np.array([(0, 0), (1, 0)], dtype=bwp.INDICES_DTYPE),
    )
    s.add_persistent_vector(
        matrix="foo",
        data_array=np.arange(10, 12),
        indices_array=np.array([(0, 1), (1, 1)], dtype=bwp.INDICES_DTYPE),
    )
    mm = MappedMatrix(packages=[s], matrix="foo", custom_filter=lambda x: x["col"] > 0)
    assert mm.matrix.sum() == 21


def test_all_empty_after_mapping():
    s = bwp.create_datapackage()
    s.add_persistent_vector(
        matrix="foo",
        data_array=np.arange(2),
        indices_array=np.array([(0, 0), (1, 0)], dtype=bwp.INDICES_DTYPE),
    )
    am = ArrayMapper(array=np.array([2, 3]))
    mm = MappedMatrix(
        packages=[s],
        matrix="foo",
        row_mapper=am,
    )

    # doesn't error out because matrix has shape from array mapper
    assert not mm.matrix.sum()
    assert mm.matrix.shape == (2, 1)
    assert not mm.matrix.tocoo().data.sum()


def test_one_empty_after_mapping():
    s = bwp.create_datapackage()
    s.add_persistent_vector(
        matrix="foo",
        data_array=np.arange(2),
        indices_array=np.array([(0, 0), (1, 0)], dtype=bwp.INDICES_DTYPE),
    )
    s.add_persistent_vector(
        matrix="foo",
        data_array=np.arange(10, 12),
        indices_array=np.array([(10, 1), (11, 1)], dtype=bwp.INDICES_DTYPE),
    )
    am = ArrayMapper(array=np.array([10, 11]))
    mm = MappedMatrix(
        packages=[s],
        matrix="foo",
        row_mapper=am,
    )
    assert mm.matrix.sum() == 21


def test_empty_combinatorial_datapackage():
    s = bwp.create_datapackage()
    s.add_persistent_vector(
        matrix="foo",
        data_array=np.arange(2),
        indices_array=np.array([(0, 0), (1, 0)], dtype=bwp.INDICES_DTYPE),
    )
    t = bwp.create_datapackage(combinatorial=True)
    t.add_persistent_array(
        matrix="bar",
        data_array=np.array([[2, 4, 8, 16]]),
        indices_array=np.array([(0, 0)], dtype=bwp.INDICES_DTYPE),
    )
    mm = MappedMatrix(
        packages=[s, t],
        matrix="foo",
    )
    for _ in range(10):
        next(mm)


def test_input_indexer_vector_raises_on_unsupported_type():
    class BadIndexer:
        index = "not-an-index"

        def __next__(self):
            return self

    s = bwp.create_datapackage(sequential=True)
    s.add_persistent_vector(
        matrix="foo",
        data_array=np.arange(3),
        indices_array=np.array([(0, 0), (1, 1), (0, 1)], dtype=bwp.INDICES_DTYPE),
    )

    mm = MappedMatrix(
        packages=[s],
        matrix="foo",
        use_arrays=False,
        use_distributions=False,
        indexer_override=BadIndexer(),
    )

    with pytest.raises(ValueError, match="Can't understand indexer value"):
        mm.input_indexer_vector()


def test_indexers_single_package():
    from matrix_utils.indexers import RandomIndexer

    dp = basic_mm(name="alpha")
    mm = MappedMatrix(packages=[dp], matrix="foo", use_arrays=False, use_distributions=False)
    gi = mm.indexers
    assert list(gi.keys()) == ["alpha"]
    assert isinstance(gi["alpha"], RandomIndexer)


def test_indexers_multiple_packages():
    from matrix_utils.indexers import RandomIndexer

    dp1 = basic_mm(name="pkg-one")
    dp2 = basic_mm(name="pkg-two")
    mm = MappedMatrix(packages=[dp1, dp2], matrix="foo", use_arrays=False, use_distributions=False)
    gi = mm.indexers
    assert set(gi.keys()) == {"pkg-one", "pkg-two"}
    assert isinstance(gi["pkg-one"], RandomIndexer)
    assert isinstance(gi["pkg-two"], RandomIndexer)
    # each package gets its own indexer instance
    assert gi["pkg-one"] is not gi["pkg-two"]


def test_local_indexers_keyed_by_group_label():
    from matrix_utils.indexers import RandomIndexer

    dp = basic_mm(name="alpha")
    mm = MappedMatrix(packages=[dp], matrix="foo", use_arrays=False, use_distributions=False)
    li = mm.local_indexers
    # basic_mm adds two vector groups: "vector" and "vector2"
    assert set(li.keys()) == {"vector", "vector2"}
    # both groups share the same package-level indexer instance
    assert isinstance(li["vector"], RandomIndexer)
    assert li["vector"] is li["vector2"]


def test_local_indexers_combinatorial_are_proxies():
    from matrix_utils.indexers import CombinatorialIndexer, Proxy

    dp = bwp.create_datapackage(combinatorial=True, name="combo")
    indices = np.array([(0, 0), (1, 1)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_array(
        matrix="foo", name="g", indices_array=indices, data_array=np.array([[1, 2], [3, 4]]).T
    )
    dp.add_persistent_array(
        matrix="foo", name="h", indices_array=indices, data_array=np.array([[5, 6], [7, 8]]).T
    )

    mm = MappedMatrix(packages=[dp], matrix="foo")
    li = mm.local_indexers
    assert set(li.keys()) == {"g", "h"}
    assert all(isinstance(v, Proxy) for v in li.values())
    # global indexer is the underlying CombinatorialIndexer
    assert isinstance(mm.indexers["combo"], CombinatorialIndexer)


def test_indexers_by_type():
    from matrix_utils.indexers import RandomIndexer, SequentialIndexer

    dp_rand = basic_mm(name="rand")
    dp_seq = bwp.create_datapackage(sequential=True, name="seq")
    dp_seq.add_persistent_vector(
        matrix="foo",
        name="v",
        indices_array=np.array([(0, 0)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([1.0]),
    )
    mm = MappedMatrix(
        packages=[dp_rand, dp_seq], matrix="foo", use_arrays=False, use_distributions=False
    )

    rand_indexers = mm.indexers_by_type(RandomIndexer)
    seq_indexers = mm.indexers_by_type(SequentialIndexer)
    assert len(rand_indexers) == 1
    assert isinstance(rand_indexers[0], RandomIndexer)
    assert len(seq_indexers) == 1
    assert isinstance(seq_indexers[0], SequentialIndexer)


def test_indexers_are_unique_true():
    dp1 = basic_mm(name="a")
    dp2 = basic_mm(name="b")
    mm = MappedMatrix(packages=[dp1, dp2], matrix="foo", use_arrays=False, use_distributions=False)
    assert mm.indexers_are_unique is True


def test_indexers_are_unique_false_with_override():
    from matrix_utils.indexers import SequentialIndexer

    shared = SequentialIndexer()
    dp1 = basic_mm(name="a")
    dp2 = basic_mm(name="b")
    mm = MappedMatrix(
        packages=[dp1, dp2],
        matrix="foo",
        use_arrays=False,
        use_distributions=False,
        indexer_override=shared,
    )
    assert mm.indexers_are_unique is False


# ── group() ───────────────────────────────────────────────────────────────────


def test_group_lookup_found():
    mm = MappedMatrix(
        packages=[basic_mm()], matrix="foo", use_arrays=False, use_distributions=False
    )
    grp = mm.group("vector")
    assert grp.label == "vector"


def test_group_lookup_not_found():
    mm = MappedMatrix(
        packages=[basic_mm()], matrix="foo", use_arrays=False, use_distributions=False
    )
    with pytest.raises(KeyError):
        mm.group("nonexistent")


# ── has_flip / has_rescale ────────────────────────────────────────────────────


def test_has_flip_true():
    mm = MappedMatrix(
        packages=[diagonal()], matrix="foo", use_arrays=False, use_distributions=False
    )
    assert mm.group("vector").has_flip is True


def test_has_flip_false():
    mm = MappedMatrix(
        packages=[basic_mm()], matrix="foo", use_arrays=False, use_distributions=False
    )
    assert mm.group("vector").has_flip is False


def test_has_rescale_false():
    mm = MappedMatrix(
        packages=[basic_mm()], matrix="foo", use_arrays=False, use_distributions=False
    )
    assert mm.group("vector").has_rescale is False


def test_has_rescale_true():
    dp = bwp.create_datapackage()
    dp.add_persistent_vector(
        matrix="foo",
        name="sv",
        indices_array=np.array([(0, 0), (1, 1)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([1.0, 2.0]),
        rescale_array=np.array([0.5, 2.0]),
    )
    mm = MappedMatrix(packages=[dp], matrix="foo", use_arrays=False, use_distributions=False)
    assert mm.group("sv").has_rescale is True


# ── has_reference / reference ──────────────────────────────────────────────────


def test_has_reference_false():
    mm = MappedMatrix(
        packages=[basic_mm()], matrix="foo", use_arrays=False, use_distributions=False
    )
    assert mm.group("vector").has_reference is False


def test_has_reference_true():
    dp = bwp.create_datapackage()
    dp.add_persistent_vector(
        matrix="foo",
        name="sv",
        indices_array=np.array([(0, 0), (1, 1)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([1.0, 2.0]),
        reference_array=np.array([True, False], dtype=bool),
    )
    mm = MappedMatrix(packages=[dp], matrix="foo", use_arrays=False, use_distributions=False)
    assert mm.group("sv").has_reference is True


def test_reference_values():
    dp = bwp.create_datapackage()
    dp.add_persistent_vector(
        matrix="foo",
        name="sv",
        indices_array=np.array([(0, 0), (1, 1)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([1.0, 2.0]),
        reference_array=np.array([True, False], dtype=bool),
    )
    mm = MappedMatrix(packages=[dp], matrix="foo", use_arrays=False, use_distributions=False)
    group = mm.group("sv")
    assert group.reference.dtype == bool
    assert list(group.reference) == [True, False]


def test_reference_current_none_when_absent():
    mm = MappedMatrix(
        packages=[basic_mm()], matrix="foo", use_arrays=False, use_distributions=False
    )
    assert mm.group("vector").reference_current is None


def test_reference_current_returns_masked_values():
    dp = bwp.create_datapackage()
    dp.add_persistent_vector(
        matrix="foo",
        name="sv",
        indices_array=np.array([(0, 0), (1, 1)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([1.0, 2.0]),
        reference_array=np.array([False, True], dtype=bool),
    )
    mm = MappedMatrix(packages=[dp], matrix="foo", use_arrays=False, use_distributions=False)
    assert list(mm.group("sv").reference_current) == [False, True]


# ── n_elements_dropped ─────────────────────────────────────────────────────────────────


def test_n_elements_dropped_none():
    mm = MappedMatrix(
        packages=[basic_mm()], matrix="foo", use_arrays=False, use_distributions=False
    )
    assert mm.n_elements_dropped == 0
    assert mm.group("vector").n_elements_dropped == 0


def test_n_elements_dropped_custom_filter():
    mm = MappedMatrix(
        packages=[basic_mm()],
        matrix="foo",
        use_arrays=False,
        use_distributions=False,
        custom_filter=lambda x: x["row"] < 5,
    )
    # basic_mm vector has rows [0, 2, 4, 8]; filter keeps [0, 2, 4], drops 8
    assert mm.group("vector").n_elements_dropped == 1


def test_n_elements_dropped_unmapped():
    # Build a matrix with a pre-existing row_mapper that excludes some ids
    dp = bwp.create_datapackage()
    dp.add_persistent_vector(
        matrix="foo",
        name="v",
        indices_array=np.array([(0, 0), (99, 0)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([1.0, 2.0]),
    )
    # row_mapper that only knows about id 0, not 99
    am = ArrayMapper(array=np.array([0]))
    mm = MappedMatrix(
        packages=[dp], matrix="foo", use_arrays=False, use_distributions=False, row_mapper=am
    )
    assert mm.group("v").n_elements_dropped == 1
    assert mm.n_elements_dropped == 1


def test_n_elements_dropped_total():
    dp1 = bwp.create_datapackage()
    dp1.add_persistent_vector(
        matrix="foo",
        name="a",
        indices_array=np.array([(0, 0), (99, 0)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([1.0, 2.0]),
    )
    dp2 = bwp.create_datapackage()
    dp2.add_persistent_vector(
        matrix="foo",
        name="b",
        indices_array=np.array([(0, 0), (98, 0)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([3.0, 4.0]),
    )
    am = ArrayMapper(array=np.array([0]))
    mm = MappedMatrix(
        packages=[dp1, dp2], matrix="foo", use_arrays=False, use_distributions=False, row_mapper=am
    )
    assert mm.n_elements_dropped == 2


# ── input_raw_indices() ───────────────────────────────────────────────────────


def test_input_raw_indices_aligns_with_data_vector():
    mm = MappedMatrix(
        packages=[basic_mm()], matrix="foo", use_arrays=False, use_distributions=False
    )
    raw = mm.input_raw_indices()
    data = mm.input_data_vector()
    assert len(raw) == len(data)


def test_input_raw_indices_contains_original_ids():
    dp = bwp.create_datapackage()
    dp.add_persistent_vector(
        matrix="foo",
        name="v",
        indices_array=np.array([(10, 20), (30, 40)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([1.0, 2.0]),
    )
    mm = MappedMatrix(packages=[dp], matrix="foo", use_arrays=False, use_distributions=False)
    raw = mm.input_raw_indices()
    assert list(raw["row"]) == [10, 30]
    assert list(raw["col"]) == [20, 40]


def test_input_raw_indices_excludes_dropped():
    dp = bwp.create_datapackage()
    dp.add_persistent_vector(
        matrix="foo",
        name="v",
        indices_array=np.array([(0, 0), (99, 0)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([1.0, 2.0]),
    )
    am = ArrayMapper(array=np.array([0]))
    mm = MappedMatrix(
        packages=[dp], matrix="foo", use_arrays=False, use_distributions=False, row_mapper=am
    )
    raw = mm.input_raw_indices()
    # id 99 was unmapped, so only id 0 survives
    assert len(raw) == 1
    assert raw["row"][0] == 0


# ── input_flip_vector() ───────────────────────────────────────────────────────


def test_input_flip_vector_no_flip():
    mm = MappedMatrix(
        packages=[basic_mm()], matrix="foo", use_arrays=False, use_distributions=False
    )
    flip = mm.input_flip_vector()
    assert flip.dtype == bool
    assert not flip.any()
    assert len(flip) == len(mm.input_data_vector())


def test_input_flip_vector_with_flip():
    mm = MappedMatrix(
        packages=[diagonal()], matrix="foo", use_arrays=False, use_distributions=False
    )
    flip = mm.input_flip_vector()
    # diagonal fixture: flip_array = [False, True, False, False]
    assert flip.dtype == bool
    assert flip.sum() == 1
    assert len(flip) == len(mm.input_data_vector())


def test_nan_in_vector_skips_insertion():
    dp = bwp.create_datapackage()
    dp.add_persistent_vector(
        matrix="foo",
        name="v",
        indices_array=np.array([(0, 0), (1, 1), (2, 2)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([1.0, np.nan, 3.0]),
    )
    mm = MappedMatrix(packages=[dp], matrix="foo", use_arrays=False, use_distributions=False)
    assert mm.matrix[0, 0] == 1.0
    assert mm.matrix[1, 1] == 0.0  # NaN skipped, stays zero
    assert mm.matrix[2, 2] == 3.0
    assert not np.isnan(mm.matrix.data).any()


def test_nan_in_array_skips_insertion():
    dp = bwp.create_datapackage(sequential=True)
    dp.add_persistent_array(
        matrix="foo",
        name="a",
        indices_array=np.array([(0, 0), (1, 1)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([[1.0, np.nan], [3.0, 4.0]]),
    )
    mm = MappedMatrix(packages=[dp], matrix="foo", use_distributions=False)
    # column 0: data = [1.0, 3.0] — both inserted
    assert mm.matrix[0, 0] == 1.0
    assert mm.matrix[1, 1] == 3.0
    next(mm)
    # column 1: data = [nan, 4.0] — first element skipped, stays zero
    assert mm.matrix[0, 0] == 0.0
    assert mm.matrix[1, 1] == 4.0
    assert not np.isnan(mm.matrix.data).any()


def test_nan_preserves_earlier_package_value():
    # Package A sets values; package B uses NaN to leave them untouched.
    dp_base = bwp.create_datapackage()
    dp_base.add_persistent_vector(
        matrix="foo",
        name="base",
        indices_array=np.array([(0, 0), (1, 1)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([5.0, 7.0]),
    )
    dp_scenario = bwp.create_datapackage()
    dp_scenario.add_persistent_vector(
        matrix="foo",
        name="scenario",
        indices_array=np.array([(0, 0), (1, 1)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([np.nan, 99.0]),
    )
    mm = MappedMatrix(
        packages=[dp_base, dp_scenario], matrix="foo", use_arrays=False, use_distributions=False
    )
    assert mm.matrix[0, 0] == 5.0  # NaN in scenario → base value preserved
    assert mm.matrix[1, 1] == 99.0  # non-NaN in scenario → overrides base
    assert not np.isnan(mm.matrix.data).any()
