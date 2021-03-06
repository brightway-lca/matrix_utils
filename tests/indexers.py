from matrix_utils import RandomIndexer, SequentialIndexer, CombinatorialIndexer, Proxy
import numpy as np


def test_random():
    a = RandomIndexer(42)
    b = RandomIndexer(42)
    assert np.allclose([next(a) for _ in range(10)], [next(b) for _ in range(10)])
    a = RandomIndexer()
    b = RandomIndexer()
    assert not np.allclose([next(a) for _ in range(10)], [next(b) for _ in range(10)])


def test_sequential():
    a = SequentialIndexer()
    assert a.index == 0
    for i in range(1, 10):
        assert next(a) == i
        assert a.index == i


def test_combinatorial():
    a = CombinatorialIndexer([4, 2, 3])
    assert a.index == (0, 0, 0)
    next(a)
    assert a.index == (0, 0, 1)
    results = [next(a) for _ in range(5)]
    expected = [
        (0, 0, 2),
        (0, 1, 0),
        (0, 1, 1),
        (0, 1, 2),
        (1, 0, 0),
    ]
    assert results == expected


def test_combinatorial_proxy():
    a = CombinatorialIndexer([4, 2, 3])
    assert a.index == (0, 0, 0)

    p = Proxy(a, 0)

    results = []
    for _ in range(5):
        results.append(p.index)
        next(a)
    expected = [0, 0, 0, 0, 0]
    assert results == expected

    a = CombinatorialIndexer([4, 2, 3])
    assert a.index == (0, 0, 0)

    p = Proxy(a, 2)

    results = []
    for _ in range(5):
        results.append(p.index)
        next(a)
    expected = [0, 1, 2, 0, 1]
    assert results == expected
