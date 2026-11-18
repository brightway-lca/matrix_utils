import bw_processing as bwp
import numpy as np
import pytest
from stats_arrays import (
    LognormalUncertainty,
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
        indices_array=np.array([(10, 10), (12, 9), (14, 8), (18, 7)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([11, 12.3, 14, 125]),
    )
    return dp


def test_basic_distributions():
    dp = mc_fixture()
    # Seeded so the statistical bounds below don't fail at random
    mm = mu.MappedMatrix(packages=[dp], matrix="foo", use_distributions=True, seed_override=42)

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

    assert 20 <= np.mean(results[row[10], col[20], :]) <= 180
    assert 5 <= np.mean(results[row[11], col[21], :]) <= 18

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


def test_distributions_reproducible_without_derived_seeds():
    dp = mc_fixture(seed=123)
    mm = mu.MappedMatrix(
        packages=[dp], matrix="foo", use_distributions=True, derive_group_seeds=False
    )

    results = np.zeros((2, 2, 10))
    results[:, :, 0] = mm.matrix.toarray()

    for i in range(1, 10):
        next(mm)
        results[:, :, i] = mm.matrix.toarray()

    given = results.sum(axis=0).sum(axis=0).ravel()
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
    mm = mu.MappedMatrix(packages=[dp], matrix="foo", use_distributions=True, seed_override=7)

    first = np.zeros((2, 2, 10))
    first[:, :, 0] = mm.matrix.toarray()

    for i in range(1, 10):
        next(mm)
        first[:, :, i] = mm.matrix.toarray()

    dp = mc_fixture(seed=567)
    mm = mu.MappedMatrix(packages=[dp], matrix="foo", use_distributions=True, seed_override=7)

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


def test_distributions_reproducible():
    dp = mc_fixture(seed=123)
    mm = mu.MappedMatrix(packages=[dp], matrix="foo", use_distributions=True)

    given = [mm.matrix.sum()]
    for _ in range(9):
        next(mm)
        given.append(mm.matrix.sum())

    # Locks in the seed derivation; must not change across platforms or releases
    expected = np.array(
        [
            36.82558217,
            30.17308861,
            30.50747818,
            22.84149632,
            25.52543653,
            30.30897259,
            24.77950859,
            27.49645739,
            33.17481873,
            28.26067776,
        ]
    )
    assert np.allclose(expected, given)


def lognormal_fixture(seed=None, name=None):
    """Two groups, each with one lognormal with identical parameters (issue #50)."""
    dp = bwp.create_datapackage(seed=seed, name=name)
    for label, row in (("a", 0), ("b", 1)):
        dp.add_persistent_vector(
            matrix="foo",
            name=label,
            indices_array=np.array([(row, row)], dtype=bwp.INDICES_DTYPE),
            data_array=np.array([1.0]),
            distributions_array=UncertaintyBase.from_dicts(
                {"loc": 0, "scale": np.log(1.5), "uncertainty_type": LognormalUncertainty.id},
            ),
        )
    return dp


def sample_diagonal(mm, n=500):
    results = np.zeros((n, 2))
    for i in range(n):
        if i:
            next(mm)
        results[i] = mm.matrix.diagonal()
    return results


def test_derive_group_seeds_removes_correlation():
    # Regression test for https://github.com/brightway-lca/matrix_utils/issues/50
    mm = mu.MappedMatrix(
        packages=[lognormal_fixture()], matrix="foo", use_distributions=True, seed_override=1
    )
    results = sample_diagonal(mm)
    assert abs(np.corrcoef(results.T)[0, 1]) < 0.2


def test_no_derive_group_seeds_gives_identical_draws():
    mm = mu.MappedMatrix(
        packages=[lognormal_fixture()],
        matrix="foo",
        use_distributions=True,
        seed_override=1,
        derive_group_seeds=False,
    )
    results = sample_diagonal(mm)
    assert np.allclose(results[:, 0], results[:, 1])


def test_derive_group_seeds_reproducible_across_package_names():
    first = mu.MappedMatrix(
        packages=[lognormal_fixture(name="one")],
        matrix="foo",
        use_distributions=True,
        seed_override=1,
    )
    second = mu.MappedMatrix(
        packages=[lognormal_fixture(name="two")],
        matrix="foo",
        use_distributions=True,
        seed_override=1,
    )
    assert [g.seed for g in first.groups] == [g.seed for g in second.groups]
    assert len({g.seed for g in first.groups}) == 2
    assert np.allclose(sample_diagonal(first, 20), sample_diagonal(second, 20))


def test_derive_group_seeds_uses_datapackage_seed():
    mm = mu.MappedMatrix(packages=[lognormal_fixture(seed=5)], matrix="foo", use_distributions=True)
    seeds = [g.seed for g in mm.groups]
    assert len(set(seeds)) == 2
    assert 5 not in seeds


def test_derive_group_seeds_repeated_group_labels():
    mm = mu.MappedMatrix(
        packages=[lognormal_fixture(), lognormal_fixture()],
        matrix="foo",
        use_distributions=True,
        seed_override=1,
    )
    assert len({g.seed for g in mm.groups}) == 4


def test_derive_group_seeds_no_seed():
    mm = mu.MappedMatrix(packages=[lognormal_fixture()], matrix="foo", use_distributions=True)
    assert [g.seed for g in mm.groups] == [None, None]


def test_derive_group_seeds_mapped_matrix_dict_keys_independent():
    # Same packages and group labels under different keys must not share seeds
    mmd = mu.MappedMatrixDict(
        packages={"x": [lognormal_fixture()], "y": [lognormal_fixture()]},
        matrix="foo",
        row_mapper=mu.ArrayMapper(array=np.array([0, 1])),
        diagonal=True,
        use_distributions=True,
        seed_override=1,
    )
    seeds = {key: [g.seed for g in mm.groups] for key, mm in mmd.items()}
    assert len(set(seeds["x"]) | set(seeds["y"])) == 4

    results = np.zeros((200, 2))
    for i in range(200):
        if i:
            next(mmd)
        results[i] = mmd["x"].matrix.diagonal()[0], mmd["y"].matrix.diagonal()[0]
    assert abs(np.corrcoef(results.T)[0, 1]) < 0.2


def test_derive_group_seeds_no_distributions_keeps_base_seed():
    mm = mu.MappedMatrix(packages=[lognormal_fixture()], matrix="foo", seed_override=1)
    assert [g.seed for g in mm.groups] == [1, 1]


def test_derive_group_seeds_sequence_seed():
    kwargs = dict(matrix="foo", use_distributions=True, seed_override=[1, 2])
    first = mu.MappedMatrix(packages=[lognormal_fixture()], **kwargs)
    second = mu.MappedMatrix(packages=[lognormal_fixture()], **kwargs)
    assert len({g.seed for g in first.groups}) == 2
    assert np.allclose(sample_diagonal(first, 20), sample_diagonal(second, 20))


@pytest.mark.parametrize("derive", [True, False])
@pytest.mark.parametrize("seed", [2**32, -1])
def test_derive_group_seeds_invalid_seed_rejected_either_way(seed, derive):
    with pytest.raises(ValueError):
        mu.MappedMatrix(
            packages=[lognormal_fixture()],
            matrix="foo",
            use_distributions=True,
            seed_override=seed,
            derive_group_seeds=derive,
        )


def test_derive_seed_labels_unambiguous():
    from matrix_utils.utils import derive_seed

    assert derive_seed(1, "a\x1fb", "c") != derive_seed(1, "a", "b\x1fc")
    assert derive_seed(1, "ab", "c") != derive_seed(1, "a", "bc")
    assert derive_seed(1, "a", "") != derive_seed(1, "a")


def test_derive_seed_deterministic():
    from matrix_utils.utils import derive_seed

    assert derive_seed(1, "foo", "a") == derive_seed(1, "foo", "a")
    assert derive_seed(1, "foo", "a") != derive_seed(1, "foo", "b")
    assert derive_seed(1, "foo", "a") != derive_seed(2, "foo", "a")
    assert 0 <= derive_seed(1, "foo", "a") < 2**32
