from .aggregation import aggregate_with_sparse
from .array_mapper import ArrayMapper
from .indexers import Indexer, Proxy
from bw_processing import DatapackageBase
from typing import Any
import numpy as np


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
    ):
        self.label = group_label
        self.package = package
        self.use_distributions = use_distributions
        self.vector = self.is_vector()
        self.aggregate = self.package.metadata["sum_intra_duplicates"]

    @property
    def data(self):
        return self.get_resource_by_suffix("data")

    @property
    def flip(self):
        return self.get_resource_by_suffix("flip")

    @property
    def indices(self):
        return self.get_resource_by_suffix("indices")

    def get_resource_by_suffix(self, suffix: str) -> Any:
        return self.package.get_resource(self.label + "." + suffix)[0]

    def is_vector(self) -> bool:
        """Determine if this is a vector or array resource"""
        metadata = self.package.get_resource(self.label + ".data")[1]
        return metadata.get('category') == 'vector'

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

    def build_mask(self, a, b):
        """Build boolean array mask where ``False`` means that a data element is not present, and should be ignored. See discussion above."""
        mask = (a != -1) * (b != -1)
        if (~mask).sum():
            return mask
        else:
            return None

    def map_indices(self, *, diagonal=False):
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
                    self.row_original[self.mask], self.col_original[self.mask], np.zeros(self.mask.sum()), self.count
                )
            else:
                self.row, self.col, _ = aggregate_with_sparse(
                    self.row_original, self.col_original, np.zeros(self.row_original.shape), self.count
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
        if vector is not None:
            data = vector
        elif self.vector:
            try:
                data = next(self.data)
            except TypeError:
                data = self.data
        else:
            data = self.data[:, self.indexer.index % self.ncols]

        data = data.copy()

        try:
            data[self.flip] *= -1
        except KeyError:
            pass

        self.current_data = data.copy()

        if self.aggregate:
            if self.mask is not None:
                return aggregate_with_sparse(
                    self.row_original[self.mask], self.col_original[self.mask], data[self.mask], self.count
                )
            else:
                return aggregate_with_sparse(
                    self.row_original, self.col_original, data, self.count
                )
        else:
            return self.row, self.col, data[self.mask]
