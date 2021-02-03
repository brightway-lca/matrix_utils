from .array_mapper import ArrayMapper
from .indexers import RandomIndexer, SequentialIndexer, CombinatorialIndexer
from .resource_group import ResourceGroup
from bw_processing import Datapackage
from scipy import sparse
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
        seed_override: Union[int, None] = None,
    ):
        self.seed_override = seed_override
        self.packages = {
            package: [
                ResourceGroup(
                    package=filtered_package,
                    group_label=group_label,
                    use_distributions=use_distributions,
                )
                for group_label, filtered_package in package.groups.items()
                if self.has_relevant_data(
                    group_label, package, use_vectors, use_arrays, use_distributions
                )
            ]
            for package in [
                obj.filter_by_attribute("matrix", matrix) for obj in packages
            ]
        }
        self.groups = tuple([obj for lst in self.packages.values() for obj in lst])
        self.add_indexers(seed_override)

        self.row_mapper = row_mapper or ArrayMapper(
                array=np.hstack([obj.unique_row_indices() for obj in self.groups]),
            )
        self.col_mapper = col_mapper or ArrayMapper(
                array=np.hstack([obj.unique_col_indices() for obj in self.groups]),
            )

        self.add_mappers(
            axis=0,
            mapper=self.row_mapper,
        )
        self.add_mappers(
            axis=1,
            mapper=self.col_mapper
        )
        self.map_indices()

        row_indices = np.hstack([obj.row for obj in self.groups])
        col_indices = np.hstack([obj.col for obj in self.groups])

        self.matrix = sparse.coo_matrix(
            (np.zeros(len(row_indices)), (row_indices, col_indices),),
            (row_indices.max() + 1, col_indices.max() + 1,),
            dtype=np.float64
        ).tocsr()

        self.rebuild_matrix()

    def add_mappers(self, axis: int, mapper: ArrayMapper):
        for group in self.groups:
            group.add_mapper(axis=axis, mapper=mapper)

    def map_indices(self):
        for group in self.groups:
            group.map_indices()

    def iterate_indexers(self):
        for obj in self.packages:
            if hasattr(obj, "indexer"):
                next(obj.indexer)

    def rebuild_matrix(self):
        self.matrix.data *= 0
        for group in self.groups:
            row, col, data = group.calculate()
            if group.package.metadata["substitute"]:
                self.matrix[row, col] = data
            else:
                self.matrix[row, col] += data

    def __next__(self):
        self.iterate_indexers()
        self.rebuild_matrix()

    def add_indexers(self, seed_override):
        """Add indexers"""
        for package, resources in self.packages.items():
            if package.metadata["combinatorial"]:
                package.indexer = CombinatorialIndexer(
                    [obj.ncols for obj in resources if obj.ncols]
                )
                for i, obj in enumerate(resources):
                    obj.add_combinatorial_indexer(indexer=package.indexer, offset=i)
            elif package.metadata["sequential"] and seed_override is None:
                package.indexer = SequentialIndexer()
                for obj in resources:
                    obj.add_indexer(indexer=package.indexer)
            else:
                package.indexer = RandomIndexer(
                    seed=seed_override or package.metadata["seed"]
                )
                for obj in resources:
                    obj.add_indexer(indexer=package.indexer)

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
