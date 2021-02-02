from .indexers import Indexer, Proxy
from .array_mapper import ArrayMapper
from .aggregation import aggregate_with_sparse
from bw_processing import DatapackageBase
import numpy as np
from typing import Any


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
        return self._get_data("data")

    @property
    def flip(self):
        return self._get_data("flip")

    @property
    def indices(self):
        return self._get_data("indices")

    def _get_data(self, suffix: str) -> Any:
        return self.package.get_resource(self.label + "." + suffix)[0]

    def is_vector(self) -> bool:
        """Determine if this is a vector or array resource"""
        return not (hasattr(self.data, "shape") and len(self.data.shape) > 1)

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
            row_disaggregated = self.row_mapper.map_array(self.indices["row"])
            col_disaggregated = self.col_mapper.map_array(self.indices["col"])
            self.row, self.col, _ = aggregate_with_sparse(
                row_disaggregated, col_disaggregated, np.zeros_like(row_disaggregated),
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

    def calculate(self):
        pass
        # return row, col, data
