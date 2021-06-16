from .aggregation import aggregate_with_sparse
from .array_mapper import ArrayMapper
from .indexers import Indexer, Proxy
from bw_processing import DatapackageBase
from stats_arrays import MCRandomNumberGenerator
from typing import Any, Union, Callable
import numpy as np


class FakeRNG:
    def __init__(self, array):
        self.array = array

    def __next__(self):
        return self.array


class ResourceGroup:
    """A class that handles a resource group - a collection of files in data package which define one matrix. A resource group can contain the following:

    * data array or interface (required)
    * indices structured array (required)
    * flip array (optional)
    * csv metadata (optional)
    * json metadata (optional)

    After instantiation, the ``MappedMatrix`` class will add an indexer (an instance of ``matrix_utils.indexers.Indexer``), using either ``add_indexer`` or ``add_combinatorial_indexer``. It will also add one or two array mappers, using ``add_mapper``. Only one mapper is needed if the matrix is diagonal.

    One easy source of confusion is the difference between ``.row_original`` and ``.row``. Both of these are one-dimensional vectors which have matrix row indices (i.e. they have already been mapped). ``.row_original`` are the values as given in the data package, whereas ``.row`` are the values actually used in matrix data insertion (and the same for column indices).There are two possible modifications that can be applied from ``.row_original`` to ``.row``; first, we can have internal aggregation, which will shrink the size of the row or column indices, as duplicate elements are eliminated. Second, if the array mapper was already instantiated, we might need to delete some elements from the row or column indices, as these values are not used in this calculation. For example, in life cycle assessment, LCIA method implementations often contain characterization factors for flows not present in the biosphere (as they were not used by an of the activities). In this case, we would need to eliminate these factors, as our characterization matrix must match exactly to the biosphere matrix already built.

    Here is an example for row indices:

    .. code-block:: python

        row_input_indices = [0, 17, 99, 42, 17]
        row_original = [0, 1, -1, 2, 1]
        after_aggregation = [0, 1, -1, 2]  # -1 is missing data point
        after_masking = [0, 1, 2]
        row = [0, 1, 2]

    Any data coming into this class, with through instantiation or via method calls such as ``.calculate``, should follow the length and order of ``.row_original``.

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
    ):
        self.label = group_label
        self.package = package
        self.use_distributions = use_distributions
        self.custom_filter = custom_filter
        self.vector = self.is_vector()

        if custom_filter is not None:
            self.filter_mask = custom_filter(self.raw_indices)
        else:
            self.filter_mask = None

        if self.use_distributions and self.vector:
            seed = seed_override or self.package.metadata.get("seed")
            if self.has_distributions:
                self.rng = MCRandomNumberGenerator(params=self.data, seed=seed)
            else:
                self.rng = FakeRNG(self.data)

        self.aggregate = self.package.metadata["sum_intra_duplicates"]
        self.empty = self.indices.shape == (0,)

    @property
    def has_distributions(self):
        try:
            self.get_resource_by_suffix("distributions")
            return True
        except KeyError:
            return False

    @property
    def data(self):
        if self.use_distributions and self.has_distributions:
            return self.get_resource_by_suffix("distributions")
        else:
            return self.get_resource_by_suffix("data")

    @property
    def raw_flip(self):
        """The source data for the flip array."""
        return self.get_resource_by_suffix("flip")

    @property
    def flip(self):
        """The flip array, with the custom filter mask applied if necessary."""
        if self.filter_mask is None:
            return self.raw_flip
        else:
            return self.raw_flip[self.filter_mask]

    @property
    def raw_indices(self):
        """The source data for the indices array."""
        return self.get_resource_by_suffix("indices")

    @property
    def indices(self):
        """The indices array, with the custom filter mask applied if necessary."""
        if self.filter_mask is None:
            return self.raw_indices
        else:
            return self.raw_indices[self.filter_mask]

    def get_resource_by_suffix(self, suffix: str) -> Any:
        return self.package.get_resource(self.label + "." + suffix)[0]

    def is_vector(self) -> bool:
        """Determine if this is a vector or array resource"""
        metadata = self.package.get_resource(self.label + ".data")[1]
        return metadata.get("category") == "vector"

    @property
    def ncols(self):
        if self.vector:
            return None
        else:
            return self.data.shape[1]

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
            self.row = np.array([])
            self.col = np.array([])
            return

        self.row_original = self.row_mapper.map_array(self.indices["row"])
        if diagonal:
            self.col_original = self.row_original
        else:
            self.col_original = self.col_mapper.map_array(self.indices["col"])

        self.mask = self.build_mask(self.row_original, self.col_original)

        if self.aggregate:
            self.count = max(self.row_original.max(), self.col_original.max()) + 1
            if self.mask is not None:
                self.row, self.col, _ = aggregate_with_sparse(
                    self.row_original[self.mask],
                    self.col_original[self.mask],
                    np.zeros(self.mask.sum()),
                    self.count,
                )
            else:
                self.row, self.col, _ = aggregate_with_sparse(
                    self.row_original,
                    self.col_original,
                    np.zeros(self.row_original.shape),
                    self.count,
                )

        else:
            if self.mask is not None:
                self.row = self.row_original[self.mask]
                self.col = self.col_original[self.mask]
            else:
                self.row = self.row_original
                self.col = self.col_original

    def unique_row_indices(self):
        """Return array of unique indices that respect aggregation policy"""
        return np.unique(self.indices["row"])

    def unique_col_indices(self):
        """Return array of unique indices that respect aggregation policy"""
        return np.unique(self.indices["col"])

    def add_indexer(self, indexer: Indexer):
        self.indexer = indexer

    def add_combinatorial_indexer(self, indexer: Indexer, offset: int):
        self.indexer = Proxy(indexer, offset)

    def calculate(self, vector: np.ndarray = None):
        """Generate row and column indices and a data vector. If ``.data`` is an iterator, draw the next value. If ``.data`` is an array, use the column given by ``.indexer``.

        ``vector`` is an optional input that overrides the data. It must be in the same order and have the same length as the data package indices (before possible aggregation and masking); see discussion above.

        """
        if self.empty:
            self.current_data = np.array([])
            return self.row, self.col, self.current_data

        if vector is not None:
            data = vector
        elif self.vector:
            if self.use_distributions:
                data = next(self.rng)
            else:
                try:
                    data = next(self.data)
                except TypeError:
                    data = self.data
        else:
            data = self.data[:, self.indexer.index % self.ncols]

        # Copy to avoid modifying original data
        data = data.copy()

        if self.filter_mask is not None:
            data = data[self.filter_mask]

        try:
            data[self.flip] *= -1
        except KeyError:
            # No flip array
            pass

        # Second copy because we want to store the data before aggregation
        self.current_data = data.copy()

        if self.aggregate:
            if self.mask is not None:
                return aggregate_with_sparse(
                    self.row_original[self.mask],
                    self.col_original[self.mask],
                    data[self.mask],
                    self.count,
                )
            else:
                return aggregate_with_sparse(
                    self.row_original, self.col_original, data, self.count
                )
        else:
            return self.row, self.col, data[self.mask]
