from typing import Any, Callable, Union

import numpy as np
from bw_processing import DatapackageBase
from stats_arrays import MCRandomNumberGenerator

from .aggregation import aggregate_with_sparse
from .array_mapper import ArrayMapper
from .indexers import Indexer, Proxy


class FakeRNG:
    def __init__(self, array):
        self.array = array

    def __next__(self):
        return self.array


def mask_array(array, mask=None):
    if mask is not None:
        return array[mask]
    else:
        return array


class ResourceGroup:
    """A class that handles a resource group - a collection of files in data package which define one matrix. A resource group can contain the following:

    * data array or interface (required)
    * indices structured array (required)
    * flip array (optional)
    * csv metadata (optional)
    * json metadata (optional)

    After instantiation, the ``MappedMatrix`` class will add an indexer (an instance of ``matrix_utils.indexers.Indexer``), using either ``add_indexer`` or ``add_combinatorial_indexer``. It will also add one or two array mappers, using ``add_mapper``. Only one mapper is needed if the matrix is diagonal.

    One easy source of confusion is the difference between ``.row_mapped`` and ``.row``. Both of these are one-dimensional vectors which have matrix row indices (i.e. they have already been mapped). ``.row_mapped`` are the values as given in the data package, whereas ``.row`` are the values actually used in matrix data insertion (and the same for column indices).There are two possible modifications that can be applied from ``.row_mapped`` to ``.row``; first, we can have internal aggregation, which will shrink the size of the row or column indices, as duplicate elements are eliminated. Second, if the array mapper was already instantiated, we might need to delete some elements from the row or column indices, as these values are not used in this calculation. For example, in life cycle assessment, LCIA method implementations often contain characterization factors for flows not present in the biosphere (as they were not used by an of the activities). In this case, we would need to eliminate these factors, as our characterization matrix must match exactly to the biosphere matrix already built.

    Here is an example for row indices:

    .. code-block:: python

        row_input_indices = [0, 17, 99, 42, 17]
        row_mapped = [0, 1, -1, 2, 1]
        after_aggregation = [0, 1, -1, 2]  # -1 is missing data point
        after_masking = [0, 1, 2]
        row = [0, 1, 2]

    Any data coming into this class, with through instantiation or via method calls such as ``.calculate``, should follow the length and order of the original datapackage data (use ``get_indices_data()`` to get the original indices.

    The current data, as entered into the matrix, is given by ``.current_data``.

    """

    def __init__(
        self,
        *,
        package: DatapackageBase,
        group_label: str,
        use_distributions: bool = False,
        seed_override: Union[int, None] = None,
        custom_filter: Union[Callable, None] = None,
        transpose: bool = False,
    ):
        self.label = group_label
        self.package = package
        self.use_distributions = use_distributions
        self.custom_filter = custom_filter
        self.transpose = transpose
        self.vector = self.is_vector()
        self.seed = seed_override or self.package.metadata.get("seed")

        if custom_filter is not None:
            self.custom_filter_mask = custom_filter(self.get_indices_data())
        else:
            self.custom_filter_mask = None

        if self.use_distributions and self.vector:
            if self.has_distributions:
                self.rng = MCRandomNumberGenerator(
                    params=self.data_original, seed=self.seed
                )
            else:
                self.rng = FakeRNG(self.data_original)

        self.aggregate = self.package.metadata["sum_intra_duplicates"]
        self.empty = self.get_indices_data().shape == (0,)

    def __str__(self):
        return "ResourceGroup {}\n\tVector: {}\n\tDistributions: {}\n\tTranspose: {}\n\tSeed: {}\n\tCustom filter: {}".format(
            self.label,
            self.vector,
            self.use_distributions,
            self.transpose,
            self.seed,
            bool(self.custom_filter),
        )

    @property
    def has_distributions(self):
        try:
            self.get_resource_by_suffix("distributions")
            return True
        except KeyError:
            return False

    @property
    def data_original(self):
        if self.use_distributions and self.has_distributions:
            return self.get_resource_by_suffix("distributions")
        else:
            return self.get_resource_by_suffix("data")

    @property
    def flip(self):
        """The flip array, with all masks applied (if necessary)."""
        return self.apply_masks(self.get_resource_by_suffix("flip"))

    def get_indices_data(self):
        """The source data for the indices array."""
        indices = self.get_resource_by_suffix("indices")
        if self.transpose:
            indices = indices.astype([("col", np.int32), ("row", np.int32)], copy=False)
        return indices

    def get_resource_by_suffix(self, suffix: str) -> Any:
        return self.package.get_resource(self.label + "." + suffix)[0]

    def is_vector(self) -> bool:
        """Determine if this is a vector or array resource"""
        metadata = self.package.get_resource(self.label + ".data")[1]
        return metadata.get("category") == "vector"

    def is_array(self) -> bool:
        """Determine if this is a vector or array resource"""
        return not self.is_vector()

    def is_interface(self) -> bool:
        """Determine if data is an interface"""
        return (
            self.package.get_resource(self.label + ".data")[1]["profile"] == "interface"
        )

    @property
    def ncols(self):
        if self.vector:
            return None
        else:
            return self.data_original.shape[1]

    def add_mapper(self, axis: int, mapper: ArrayMapper):
        if axis == 0:
            self.row_mapper = mapper
        else:
            self.col_mapper = mapper

    def build_mask(self, row, col):
        """Build boolean array mask where ``False`` means that a data element is not present, and should be ignored. See discussion above."""
        mask = (row != -1) * (col != -1)
        if (~mask).sum():
            return mask
        else:
            return None

    def map_indices(self, *, diagonal=False):
        if self.empty:
            self.row_matrix = self.row_masked = np.array([], dtype=np.int64)
            self.col_matrix = self.col_masked = np.array([], dtype=np.int64)
            return

        indices = self.get_indices_data()

        self.row_mapped = self.row_mapper.map_array(indices["row"])
        if diagonal:
            self.col_mapped = self.row_mapped
        else:
            self.col_mapped = self.col_mapper.map_array(indices["col"])

        self.unmapped_mask = self.build_mask(
            mask_array(self.row_mapped, self.custom_filter_mask),
            mask_array(self.col_mapped, self.custom_filter_mask),
        )

        self.row_masked = self.apply_masks(self.row_mapped)
        self.col_masked = self.apply_masks(self.col_mapped)

        if self.row_masked.shape == (0,):
            self.row_matrix = self.row_masked
            self.col_matrix = self.col_masked
            self.empty = True
            return

        if self.aggregate:
            self.count = max(self.row_masked.max(), self.col_masked.max()) + 1
            self.row_matrix, self.col_matrix, _ = aggregate_with_sparse(
                self.row_masked,
                self.col_masked,
                np.zeros(self.row_masked.shape),
                self.count,
            )
        else:
            self.row_matrix = self.row_masked
            self.col_matrix = self.col_masked

    def unique_row_indices_for_mapping(self):
        """Return array of unique indices that respect aggregation policy"""
        return np.unique(
            mask_array(self.get_indices_data()["row"], self.custom_filter_mask)
        )

    def unique_col_indices_for_mapping(self):
        """Return array of unique indices that respect aggregation policy"""
        return np.unique(
            mask_array(self.get_indices_data()["col"], self.custom_filter_mask)
        )

    def add_indexer(self, indexer: Indexer):
        self.indexer = indexer

    def add_combinatorial_indexer(self, indexer: Indexer, offset: int):
        self.indexer = Proxy(indexer, offset)

    def apply_masks(self, array):
        """Apply both ``custom_filter_mask`` (if present) and ``unmapped_mask``."""
        if self.custom_filter_mask is not None:
            array = array[self.custom_filter_mask]

        if self.unmapped_mask is not None:
            array = array[self.unmapped_mask]

        return array

    def calculate(self, vector: np.ndarray = None):
        """Generate row and column indices and a data vector. If ``.data`` is an iterator, draw the next value. If ``.data`` is an array, use the column given by ``.indexer``.

        ``vector`` is an optional input that overrides the data. It must be in the same order and have the same length as the data package indices (before possible aggregation and masking); see discussion above.

        """
        if self.empty:
            self.data_current = np.array([])
            return self.row_matrix, self.col_matrix, self.data_current

        if vector is not None:
            data = vector
        elif self.vector:
            if self.use_distributions and self.has_distributions:
                data = next(self.rng)
            else:
                try:
                    data = next(self.data_original)
                except TypeError:
                    data = self.data_original
        else:
            data = self.data_original[:, self.indexer.index % self.ncols]

        # `data` is now in the original state before **any** masking`
        # Copy to avoid modifying original data
        data = data.copy()

        # apply both custom filter mask and mapping mask
        data = self.apply_masks(data)

        try:
            data[self.flip] *= -1
        except KeyError:
            # No flip array
            pass

        # Second copy because we want to store the data before aggregation
        self.data_current = data.copy()

        if self.aggregate:
            return aggregate_with_sparse(
                self.row_masked,
                self.col_masked,
                data,
                self.count,
            )
        else:
            return self.row_matrix, self.col_matrix, data
