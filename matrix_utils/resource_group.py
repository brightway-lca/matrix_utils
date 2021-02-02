from bw_processing import DatapackageBase
import numpy as np
from .utils import get_exactly_one


class ResourceGroup:
    def __init__(
        self,
        *,
        package: DatapackageBase,
        group_label: str,
        use_vectors: bool = True,
        use_arrays: bool = True,
        use_distributions: bool = False,
    ):
        self.label = group_label
        self.package = package

        self.use_vectors = use_vectors
        self.use_arrays = use_arrays
        self.use_distributions = use_distributions

    def _get_indices(self):
        if not hasattr(self, "values"):
            self.values = get_exactly_one(
                self.package.get_resource(o['name'])[0] for o in self.package.resources if o["kind"] == "indices"
            )

    def unique_row_indices(self):
        self._get_indices()
        print(type(self.values))
        print(self.values)
        return np.unique(self.values["row"])

    def unique_col_indices(self):
        self._get_indices()
        return np.unique(self.values["col"])

    def map_row_indices(self, mapper):
        pass

    def map_col_indices(self, mapper):
        pass
