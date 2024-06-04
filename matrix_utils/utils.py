import numpy as np

from .errors import AllArraysEmpty


def filter_groups_for_packages(
    packages, matrix_label, use_vectors, use_arrays, use_distributions
) -> dict:
    return {
        package: [
            (group_label, filtered_package)
            for group_label, filtered_package in package.groups.items()
            if has_relevant_data(group_label, package, use_vectors, use_arrays, use_distributions)
        ]
        for package in [obj.filter_by_attribute("matrix", matrix_label) for obj in packages]
    }


def has_relevant_data(group_label, package, use_vectors, use_arrays, use_distributions) -> bool:
    return any(
        res
        for res in package.resources
        if res["group"] == group_label
        and (
            (res["kind"] == "data" and res["category"] == "vector" and use_vectors)
            or (
                res["kind"] == "distributions" and res["category"] == "vector" and use_distributions
            )
            # Use vectors under Monte Carlo as fallback. Warning: Could be changed in future!
            or (res["kind"] == "data" and res["category"] == "vector" and use_distributions)
            or (res["kind"] == "data" and res["category"] == "array" and use_arrays)
        )
    )


def safe_concatenate_indices(arrays: [np.ndarray], empty_ok: bool = False) -> np.ndarray:
    try:
        return np.hstack(arrays)
    except ValueError:
        if empty_ok:
            return np.array([], dtype=int)
        else:
            raise AllArraysEmpty


def unroll(a: tuple, b: tuple) -> tuple:
    """Create a new tuple combining `a` and `b`, but inline `a` or `b` if they include tuples."""
    try:
        t1 = any(isinstance(elem, tuple) for elem in a)
    except TypeError:
        t1 = False
    try:
        t2 = any(isinstance(elem, tuple) for elem in b)
    except TypeError:
        t2 = False

    if t1 and t2:
        return (*a, *b)
    elif t1:
        return (*a, b)
    elif t2:
        return (a, *b)
    else:
        return (a, b)
