from .resource_group import ResourceGroup
from .array_mapper import ArrayMapper
from bw_processing import Datapackage
from typing import Union, Sequence
import numpy as np
from scipy import sparse


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
        self.packages = {
            package: [
                ResourceGroup(package=filtered_package, group_label=group_label)
                for group_label, filtered_package in package.groups.items()
                if self.has_relevant_data(
                    group_label, package, use_vectors, use_arrays, use_distributions
                )
            ]
            for package in [
                obj.filter_by_attribute("matrix", matrix) for obj in packages
            ]
        }
        self.groups = [obj for lst in self.packages.values() for obj in lst]
        self.add_indexers()

        self.row_mapper = row_mapper or ArrayMapper(
            array=np.hstack([obj.unique_row_indices() for obj in self.groups]),
        )
        self.col_mapper = col_mapper or ArrayMapper(
            array=np.hstack([obj.unique_col_indices() for obj in self.groups]),
        )

        self.row_indices = [
            self.row_mapper.map_array(obj.row_indices()) for obj in self.groups
        ]
        self.col_indices = [
            self.col_mapper.map_array(obj.col_indices()) for obj in self.groups
        ]

        self.matrix = sparse.coo_matrix(
            (
                np.zeros_like(np.hstack(self.row_indices)),
                (np.hstack(self.row_indices), np.hstack(self.col_indices)),
            ),
            (
                max(o.max() for o in self.row_indices) + 1,
                max(o.max() for o in self.col_indices) + 1,
            ),
        )

        # self.rebuild_matrix()

    def add_indexers(self):
        """Add indexers"""
        pass

    # def set_index(self, index: int) -> None:
    #     """Override the given index in all indexers."""
    #     for group in self.groups:
    #         group.set_index(index)

    def has_relevant_data(
        self, group_label, package, use_vectors, use_arrays, use_distributions
    ):
        return any(
            res
            for res in package.resources
            if res["group"] == group_label
            and (res["kind"] == "data" and res["category"] == "vector" and use_vectors)
            or (
                res["kind"] == "distributions"
                and res["category"] == "vector"
                and use_distributions
            )
            or (res["kind"] == "data" and res["category"] == "array" and use_arrays)
        )
