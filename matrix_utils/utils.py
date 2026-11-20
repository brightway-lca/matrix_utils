import hashlib
from typing import Optional
from warnings import warn

import numpy as np
from bw_processing import UNCERTAINTY_DTYPE

from matrix_utils.errors import AllArraysEmpty


def validate_seed(seed) -> None:
    """Raise ``ValueError`` (or ``TypeError``) unless ``seed`` is a valid RNG seed.

    Valid seeds are what ``numpy.random.RandomState`` accepts: an integer in
    ``[0, 2**32)``, or a sequence of such integers. We check this ourselves instead of
    relying on ``stats_arrays``, whose seed handling has changed between releases."""
    np.random.RandomState(seed)


def as_uncertainty_array(array: np.ndarray) -> np.ndarray:
    """Return ``array`` with the dtype ``bw_processing.UNCERTAINTY_DTYPE``.

    Accepts structured arrays with the same fields in any order and with any float
    width, e.g. ``float32`` fields (``bw_processing`` before 2.0) or the field order of
    ``stats_arrays.UncertaintyBase.from_dicts``. Fields are copied by name, as
    ``ndarray.astype`` would match structured fields by position.

    Raises ``TypeError`` if the field names differ."""
    target = np.dtype(UNCERTAINTY_DTYPE)
    if array.dtype == target:
        return array
    if array.dtype.names is None or set(array.dtype.names) != set(target.names):
        raise TypeError(
            "Distributions datatype should have the fields of "
            "`bw_processing.UNCERTAINTY_DTYPE`, but was {}".format(array.dtype)
        )
    result = np.empty(array.shape, dtype=target)
    for field in target.names:
        result[field] = array[field]
    return result


def derive_seed(seed, *labels: str) -> int:
    """Deterministically derive a new RNG seed from ``seed`` and some string ``labels``.

    ``seed`` can be anything ``numpy.random.RandomState`` accepts: an integer in
    ``[0, 2**32)``, or a sequence of such integers. Invalid seeds raise the same error
    as ``RandomState`` would, so a given seed is accepted or rejected regardless of
    whether seeds are derived.

    The same inputs always give the same output, across Python processes and platforms
    (we use ``hashlib`` instead of the salted builtin ``hash``). Different labels give
    statistically independent seeds via ``numpy.random.SeedSequence``. Labels are
    length-prefixed before hashing, so e.g. ``("a_b", "c")`` and ``("a", "b_c")`` can't
    collide whatever the separator.

    Returns an integer in ``[0, 2**32)``, as required by ``numpy.random.RandomState``.
    """
    validate_seed(seed)
    seed_entropy = [int(x) for x in np.atleast_1d(seed)]

    hasher = hashlib.sha256()
    for label in labels:
        encoded = label.encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "little"))
        hasher.update(encoded)
    label_entropy = int.from_bytes(hasher.digest()[:16], "little")

    return int(np.random.SeedSequence(seed_entropy + [label_entropy]).generate_state(1)[0])


def filter_groups_for_packages(
    packages, matrix_label, use_vectors, use_arrays, use_distributions
) -> dict:
    result = {}
    for obj in packages:
        package = obj.filter_by_attribute("matrix", matrix_label)
        warn_ungrouped_resources(package, matrix_label)
        result[package] = [
            (group_label, filtered_package)
            for group_label, filtered_package in package.groups.items()
            if has_relevant_data(
                group_label, filtered_package, use_vectors, use_arrays, use_distributions
            )
        ]
    return result


def warn_ungrouped_resources(package, matrix_label) -> None:
    """Warn about resources for ``matrix_label`` that have no group label.

    These can't be part of a resource group, so they are ignored when building the
    matrix."""
    # Same test as ``Datapackage.groups``, which skips these resources
    ungrouped = [res.get("name") for res in package.resources if not res.get("group")]
    if ungrouped:
        warn(
            "Ignoring {} resource(s) for matrix '{}' without a group label: {}".format(
                len(ungrouped), matrix_label, ungrouped
            ),
            UserWarning,
            # Point at the code that created the `MappedMatrix`
            stacklevel=4,
        )


def has_relevant_data(group_label, package, use_vectors, use_arrays, use_distributions) -> bool:
    """Check whether resource group ``group_label`` in ``package`` has data we can use.

    ``package`` can be a whole datapackage, or one already filtered to that group (as
    in ``filter_groups_for_packages``, which avoids scanning every resource of the
    package for each group). Resources without a group label are ignored."""
    return any(
        res
        for res in package.resources
        if res.get("group") == group_label
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
