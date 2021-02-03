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
        self.sum_duplicates = self.package.metadata["sum_duplicates"]
        self.vector = self.is_vector()

    @property
    def data(self):
        return self._get_resource("data")

    @property
    def flip(self):
        return self._get_resource("flip")

    @property
    def indices(self):
        return self._get_resource("indices")

    def _get_resource(self, suffix: str) -> Any:
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

    def map_indices(self):
        if self.sum_duplicates:
            self.row_disaggregated = self.row_mapper.map_array(self.indices["row"])
            self.col_disaggregated = self.col_mapper.map_array(self.indices["col"])
            self.count = max(self.row_disaggregated.max(), self.col_disaggregated.max()) + 1
            self.row, self.col, _ = aggregate_with_sparse(
                self.row_disaggregated, self.col_disaggregated, np.zeros(len(self.row_disaggregated)), self.count
            )
        else:
            self.row = self.row_mapper.map_array(self.indices["row"])
            self.col = self.col_mapper.map_array(self.indices["col"])

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
            data = self.data[:, self.indexer.index]

        try:
            data[self.flip] *= -1
        except KeyError:
            pass

        self.sample = data.copy()

        if self.sum_duplicates:
            return aggregate_with_sparse(
                self.row_disaggregated, self.col_disaggregated, data, self.count
            )
        else:
            return self.row, self.col, data
