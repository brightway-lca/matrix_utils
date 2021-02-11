from .aggregation import aggregate_with_sparse
from .array_mapper import ArrayMapper
from .indexers import Indexer, Proxy
from bw_processing import DatapackageBase
from typing import Any
import numpy as np


class ResourceGroup:
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

        self.sample = data.copy()

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
