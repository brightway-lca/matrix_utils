from bw_processing import DatapackageBase
import numpy as np
from .utils import get_exactly_one


# combinatorial: bool = False,
# sequential: bool = False,
# seed: Union[int, None] = None,
# sum_duplicates: bool = False,
# substitute: bool = True,


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
                self.package.get_resource(o["name"])[0]
                for o in self.package.resources
                if o["kind"] == "indices"
            )

    def row_indices(self):
        self._get_indices()
        return self.values["row"]

    def col_indices(self):
        self._get_indices()
        return self.values["col"]

    def unique_row_indices(self):
        return np.unique(self.row_indices())

    def unique_col_indices(self):
        return np.unique(self.col_indices())

    def map_row_indices(self, mapper):
        pass

    def map_col_indices(self, mapper):
        pass
