from .resource_group import ResourceGroup
from .array_mapper import ArrayMapper
from bw_processing import Datapackage
from typing import Union, Sequence
import numpy as np


class MappedMatrix:
    """A scipy sparse matrix handler which takes in ``bw_processing`` data packages. Row and column values are mapped to indices, and a matrix is constructed."""

    def __init__(
        self,
        *,
        packages: Sequence[Datapackage],
        matrix: str,
        use_vectors: bool = True,
        use_arrays: bool = True,
        use_distributions: bool = False,
        row_mapper: Union[ArrayMapper, None] = None,
        col_mapper: Union[ArrayMapper, None] = None,
        seed_override: Union[int, None] = None
    ):
        self.packages = [
            obj.filter_by_attribute("matrix", matrix)
            for obj in packages
        ]
        self.groups = [
            ResourceGroup(package=package, group_label=group_label)
            for obj in self.packages
            for group_label, package in obj.groups.items()
            if self.has_relevant_data(group_label, package, use_vectors, use_arrays, use_distributions)
        ]

        self.row_mapper = row_mapper
        if self.row_mapper is None:
            self.row_mapper = ArrayMapper(
                array=np.hstack([obj.unique_row_indices() for obj in self.groups]),
            )

        self.col_mapper = col_mapper
        if self.col_mapper is None:
            self.col_mapper = ArrayMapper(
                array=np.hstack([obj.unique_col_indices() for obj in self.groups]),
            )

        # self.create_base_matrix()

    def has_relevant_data(self, group_label, package, use_vectors, use_arrays, use_distributions):
        return any((
            any(res for res in package.resources if res['kind'] == 'data' and res['category'] == 'vector' and use_vectors),
            any(res for res in package.resources if res['kind'] == 'distributions' and res['category'] == 'vector' and use_distributions),
            any(res for res in package.resources if res['kind'] == 'data' and res['category'] == 'array' and use_arrays),
        ))

    def create_base_matrix(self):
        """Create the sparse matrix structure with zeros as data values"""
        pass
