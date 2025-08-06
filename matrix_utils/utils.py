from typing import Optional

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


def handle_all_arrays_empty(
    packages: dict, matrix_label: str, identifier: Optional[str] = None
) -> None:
    """Format the error messages for `AllArraysEmpty` to make them understandable"""
    if not packages and not identifier:
        ERROR = """
No data found to build {} matrix.

No datapackages found which could provide data to build this matrix.
""".format(
            matrix_label
        )
        raise AllArraysEmpty(ERROR)
    elif not packages:
        ERROR = """
No data found to build {} matrix for {}.

No datapackages found which could provide data to build this matrix.
""".format(
            matrix_label, identifier
        )
        raise AllArraysEmpty(ERROR)
    else:
        ERROR_START = """
No data found to build {} matrix.

This error commonly occurs when using impact assessment methods for the wrong version of the
background database, because each background database version has its own set of elementary flows.

Found {} resource groups in {} datapackages but none of them had data for the requested method:
""".format(
            matrix_label, len(packages), sum(len(group) for group in packages.values())
        )
        ERROR_DETAIL_TEMPLATE = """
Datapackage name: {}
Resource group: {}
Data array length: {} (none of this data could be used)
"""
        ERROR_DETAILS = [
            ERROR_DETAIL_TEMPLATE.format(
                package.metadata["name"], group.identifier or group.label, len(group.data_original)
            )
            for package, groups in packages.items()
            for group in groups
        ]
        raise AllArraysEmpty(ERROR_START + "".join(ERROR_DETAILS))
