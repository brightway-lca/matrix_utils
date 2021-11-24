import bw_processing as bwp
import numpy as np
import pytest
from stats_arrays import (
    NormalUncertainty,
    TriangularUncertainty,
    UncertaintyBase,
    UniformUncertainty,
)

import matrix_utils as mu


def mc_fixture(**kwargs):
    dp = bwp.create_datapackage(**kwargs)
    dp.add_persistent_vector(
        matrix="foo",
        name="first",
        indices_array=np.array([(0, 0), (0, 1)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([1, 2]),
        distributions_array=UncertaintyBase.from_dicts(
            {"loc": 0, "scale": 0.5, "uncertainty_type": NormalUncertainty.id},
            {
                "loc": 4,
                "minimum": 2,
                "maximum": 10,
                "uncertainty_type": TriangularUncertainty.id,
            },
        ),
    )
    dp.add_persistent_vector(
        matrix="foo",
        name="second",
        indices_array=np.array([(1, 0), (1, 1)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([11, 12.3]),
        distributions_array=UncertaintyBase.from_dicts(
            {"loc": 10, "scale": 0.5, "uncertainty_type": NormalUncertainty.id},
            {
                "loc": 15,
                "minimum": 0,
                "maximum": 20,
                "uncertainty_type": TriangularUncertainty.id,
            },
        ),
    )
    dp.add_persistent_vector(
        matrix="bar",
        name="third",
        indices_array=np.array([(10, 20), (11, 21)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([12, 34]),
        # PDFs can be completely different from "best guess" values
        distributions_array=UncertaintyBase.from_dicts(
            {
                "loc": 100,
                "minimum": 0,
                "maximum": 200,
                "uncertainty_type": UniformUncertainty.id,
            },
            {
                "loc": 15,
                "minimum": 0,
                "maximum": 20,
                "uncertainty_type": TriangularUncertainty.id,
            },
        ),
    )
    # No uncertainties present
    dp.add_persistent_vector(
        matrix="bar",
        name="fourth",
        indices_array=np.array(
            [(10, 10), (12, 9), (14, 8), (18, 7)], dtype=bwp.INDICES_DTYPE
        ),
        data_array=np.array([11, 12.3, 14, 125]),
    )
    return dp


def test_basic_distributions():
    dp = mc_fixture()
    mm = mu.MappedMatrix(packages=[dp], matrix="foo", use_distributions=True)

    results = np.zeros((2, 2, 10))
    results[:, :, 0] = mm.matrix.toarray()

    for i in range(1, 10):
        next(mm)
        results[:, :, i] = mm.matrix.toarray()

    row = mm.row_mapper.to_dict()
    col = mm.col_mapper.to_dict()

    assert -1 <= np.mean(results[row[0], col[0], :]) <= 1
    assert 0.25 < np.std(results[row[0], col[0], :]) < 0.75

    assert results[row[1], col[1], :].min() >= 0
    assert results[row[1], col[1], :].min() <= 20
    # mean is 35/3
    assert 7.5 < np.mean(results[row[1], col[1], :]) < 16.5
    assert np.unique(results).shape == (40,)


def test_distributions_without_uncertainties():
    dp = mc_fixture()
    mm = mu.MappedMatrix(packages=[dp], matrix="bar", use_distributions=True)

    results = np.zeros((5, 6, 10))
    results[:, :, 0] = mm.matrix.toarray()

    for i in range(1, 10):
        next(mm)
        results[:, :, i] = mm.matrix.toarray()

    row = mm.row_mapper.to_dict()
    col = mm.col_mapper.to_dict()

    assert 50 <= np.mean(results[row[10], col[20], :]) <= 150
    assert 8 <= np.mean(results[row[11], col[21], :]) <= 14

    assert np.allclose(results[row[10], col[10], :], 11)
    assert np.allclose(results[row[18], col[7], :], 125)
    # Zero plus 4 fixed plus 2 variable over 10 iterations
    assert np.unique(results).shape == (25,)


def test_distributions_not_allowed():
    dp = mc_fixture()
    mm = mu.MappedMatrix(packages=[dp], matrix="foo", use_distributions=False)

    results = np.zeros((2, 2, 10))
    results[:, :, 0] = mm.matrix.toarray()

    for i in range(1, 10):
        next(mm)
        results[:, :, i] = mm.matrix.toarray()

    assert np.unique(results).shape == (4,)


def test_distributions_seed_in_datapackage():
    dp = mc_fixture(seed=123)
    mm = mu.MappedMatrix(packages=[dp], matrix="foo", use_distributions=True)

    results = np.zeros((2, 2, 10))
    results[:, :, 0] = mm.matrix.toarray()

    for i in range(1, 10):
        next(mm)
        results[:, :, i] = mm.matrix.toarray()

    dp = mc_fixture(seed=123)
    mm = mu.MappedMatrix(packages=[dp], matrix="foo", use_distributions=True)

    other = np.zeros((2, 2, 10))
    other[:, :, 0] = mm.matrix.toarray()

    for i in range(1, 10):
        next(mm)
        other[:, :, i] = mm.matrix.toarray()

    assert np.allclose(results, other)


def test_distributions_reproducible():
    dp = mc_fixture(seed=123)
    mm = mu.MappedMatrix(packages=[dp], matrix="foo", use_distributions=True)

    results = np.zeros((2, 2, 10))
    results[:, :, 0] = mm.matrix.toarray()

    for i in range(1, 10):
        next(mm)
        results[:, :, i] = mm.matrix.toarray()

    given = results.sum(axis=0).sum(axis=0).ravel()
    print(given.shape)
    print(given)
    expected = np.array(
        [
            21.06909828,
            29.21713645,
            37.0735732,
            32.0954309,
            22.10498503,
            30.75377088,
            24.87344966,
            31.2384641,
            27.44208816,
            27.45639759,
        ]
    )
    assert np.allclose(expected, given)


def test_distributions_seed_override():
    dp = mc_fixture(seed=123)
    mm = mu.MappedMatrix(packages=[dp], matrix="foo", use_distributions=True)

    results = np.zeros((2, 2, 10))
    results[:, :, 0] = mm.matrix.toarray()

    for i in range(1, 10):
        next(mm)
        results[:, :, i] = mm.matrix.toarray()

    dp = mc_fixture(seed=123)
    mm = mu.MappedMatrix(
        packages=[dp], matrix="foo", use_distributions=True, seed_override=7
    )

    first = np.zeros((2, 2, 10))
    first[:, :, 0] = mm.matrix.toarray()

    for i in range(1, 10):
        next(mm)
        first[:, :, i] = mm.matrix.toarray()

    dp = mc_fixture(seed=567)
    mm = mu.MappedMatrix(
        packages=[dp], matrix="foo", use_distributions=True, seed_override=7
    )

    second = np.zeros((2, 2, 10))
    second[:, :, 0] = mm.matrix.toarray()

    for i in range(1, 10):
        next(mm)
        second[:, :, i] = mm.matrix.toarray()

    assert not np.allclose(results, first)
    assert np.allclose(first, second)


def test_distributions_only_array_present():
    dp = bwp.create_datapackage(sequential=True)
    dp.add_persistent_array(
        matrix="foo",
        name="first",
        indices_array=np.array([(0, 0), (0, 1)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([[1, 2], [3, 4]]),
    )
    mm = mu.MappedMatrix(
        packages=[dp],
        matrix="foo",
        use_distributions=True,
        use_arrays=True,
    )
    assert mm.matrix.sum() == 4
    next(mm)
    assert mm.matrix.sum() == 6
    next(mm)
    assert mm.matrix.sum() == 4
