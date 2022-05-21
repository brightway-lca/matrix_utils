from pathlib import Path

import bw_processing as bwp
import numpy as np
import pytest
from fixtures import basic_mm, diagonal
from fs.zipfs import ZipFS

from matrix_utils import MappedMatrix, ArrayMapper
from matrix_utils.errors import AllArraysEmpty, EmptyArray

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
    with pytest.raises(AllArraysEmpty):
        MappedMatrix(
            packages=[],
            matrix="foo",
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

    dp_1 = bwp.load_datapackage(ZipFS(dirpath / "sa-1.zip"))
    dp_1.rehydrate_interface("a", ArrayInterface())

    dp_2 = bwp.load_datapackage(ZipFS(dirpath / "sa-2.zip"))
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
            (99, magic % 100, np.NaN, np.NaN, np.NaN, np.NaN, False),
            (99, magic % 100, np.NaN, np.NaN, np.NaN, np.NaN, False),
            (99, magic % 100, np.NaN, np.NaN, np.NaN, np.NaN, False),
            (98, -8.5, 1.118034, np.NaN, np.NaN, np.NaN, False),
            (98, 9.5, 1.118034, np.NaN, np.NaN, np.NaN, False),
            (4, 1, np.NaN, np.NaN, 0.5, 1.5, False),
            (4, 2, np.NaN, np.NaN, 1.5, 2.5, False),
            (0, 1, np.NaN, np.NaN, np.NaN, np.NaN, True),
            (4, 3, np.NaN, np.NaN, 2.5, 3.5, True),
            (99, 1, np.NaN, np.NaN, np.NaN, np.NaN, False),
            (99, 2, np.NaN, np.NaN, np.NaN, np.NaN, False),
            (99, 3, np.NaN, np.NaN, np.NaN, np.NaN, False),
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
            (99, magic % 100, np.NaN, np.NaN, np.NaN, np.NaN, False),
            (99, magic % 100, np.NaN, np.NaN, np.NaN, np.NaN, False),
            (99, magic % 100, np.NaN, np.NaN, np.NaN, np.NaN, False),
            (98, -8, 0.8164966, np.NaN, np.NaN, np.NaN, False),
            (98, 9, 0.8164966, np.NaN, np.NaN, np.NaN, False),
            (4, 1, np.NaN, np.NaN, 0.5, 1.5, False),
            (4, 2, np.NaN, np.NaN, 1.5, 2.5, False),
            (0, 1, np.NaN, np.NaN, np.NaN, np.NaN, True),
            (4, 3, np.NaN, np.NaN, 2.5, 3.5, True),
            (99, 1, np.NaN, np.NaN, np.NaN, np.NaN, False),
            (99, 2, np.NaN, np.NaN, np.NaN, np.NaN, False),
            (99, 3, np.NaN, np.NaN, np.NaN, np.NaN, False),
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
            (99, magic % 100, np.NaN, np.NaN, np.NaN, np.NaN, False),
            (99, magic % 100, np.NaN, np.NaN, np.NaN, np.NaN, False),
            (99, magic % 100, np.NaN, np.NaN, np.NaN, np.NaN, False),
            (98, -8.5, 1.118034, np.NaN, np.NaN, np.NaN, False),
            (98, 9.5, 1.118034, np.NaN, np.NaN, np.NaN, False),
            (0, 1, np.NaN, np.NaN, np.NaN, np.NaN, False),
            (0, 2, np.NaN, np.NaN, np.NaN, np.NaN, False),
            (0, -1, np.NaN, np.NaN, np.NaN, np.NaN, False),
            (0, -3, np.NaN, np.NaN, np.NaN, np.NaN, False),
            (99, 1, np.NaN, np.NaN, np.NaN, np.NaN, False),
            (99, 2, np.NaN, np.NaN, np.NaN, np.NaN, False),
            (99, 3, np.NaN, np.NaN, np.NaN, np.NaN, False),
        ],
        dtype=bwp.UNCERTAINTY_DTYPE,
    )
    ua = mm.input_uncertainties()
    for field, _ in bwp.UNCERTAINTY_DTYPE:
        assert np.allclose(ua[field], expected[field], equal_nan=True)


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
        print(mm.input_indexer_vector())
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
        MappedMatrix(
            packages=[s],
            matrix="foo",
            custom_filter=lambda x: x['col'] > 0
        )

    assert MappedMatrix(
        packages=[s],
        matrix="foo",
        custom_filter=lambda x: x['col'] > 0,
        empty_ok=True
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
    mm = MappedMatrix(
        packages=[s],
        matrix="foo",
        custom_filter=lambda x: x['col'] > 0
    )
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
        matrix='foo',
    )
    for _ in range(10):
        next(mm)
