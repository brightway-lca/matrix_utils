from .array_mapper import ArrayMapper
from .indexers import RandomIndexer, SequentialIndexer, CombinatorialIndexer
from .resource_group import ResourceGroup
from .utils import filter_groups_for_packages, safe_concatenate_indices
from bw_processing import Datapackage
from scipy import sparse
from typing import Union, Sequence, Any, Callable
import numpy as np


class MappedMatrix:
    """A scipy sparse matrix handler which takes in ``bw_processing`` data packages. Row and column ids are mapped to matrix indices, and a matrix is constructed.

    `indexer_override` allows for custom indexer behaviour. Indexers should follow a simple API: they must support `.__next__()`, and have the attribute `.index`, which returns an integer.

    `custom_filter` allows you to remove some data based on their indices. It is applied to all resource groups. If you need more fine-grained control, process the matrix after construction/iteration. `custom_filter` should take the indices array as an input, and return a Numpy boolean array with the same length as the indices array.

    Args:

        * packages: A list of Ddatapackage objects.
        * matrix: The string identifying the matrix to be built.
        * use_vectors: Flag to use vector data from datapackages
        * use_arrays: Flag to use array data from datapackages
        * use_distributions: Flag to use `stats_arrays` distribution data from datapackages
        * row_mapper: Optional instance of `ArrayMapper`. Used when matrices must align.
        * col_mapper: Optional instance of `ArrayMapper`. Used when matrices must align.
        * seed_override: Optional integer. Overrides the RNG seed given in the datapackage, if any.
        * indexer_override: Parameter for custom indexers. See above.
        * diagonal: If True, only use the `row` indices to build a diagonal matrix.
        * custom_filter: Callable for function to filter data based on `indices` values. See above.
        * empty_ok: If False, raise `AllArraysEmpty` if the matrix would be empty

    """

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
        indexer_override: Any = None,
        diagonal: bool = False,
        custom_filter: Union[Callable, None] = None,
        empty_ok: bool = False,
    ):
        self.seed_override = seed_override
        self.diagonal = diagonal
        self.matrix_label = matrix
        self.packages = {
            package: [
                ResourceGroup(
                    package=filtered_package,
                    group_label=group_label,
                    use_distributions=use_distributions,
                    seed_override=seed_override,
                    custom_filter=custom_filter,
                )
                for group_label, filtered_package in lst
            ]
            for package, lst in filter_groups_for_packages(
                packages, matrix, use_vectors, use_arrays, use_distributions
            ).items()
        }
        self.groups = tuple([obj for lst in self.packages.values() for obj in lst])
        self.add_indexers(indexer_override, seed_override)

        self.row_mapper = row_mapper or ArrayMapper(
            array=safe_concatenate_indices(
                [obj.unique_row_indices() for obj in self.groups], empty_ok
            ),
            empty_ok=empty_ok,
        )
        if not diagonal:
            self.col_mapper = col_mapper or ArrayMapper(
                array=safe_concatenate_indices(
                    [obj.unique_col_indices() for obj in self.groups], empty_ok
                ),
                empty_ok=empty_ok,
            )

        self.add_mappers(
            axis=0,
            mapper=self.row_mapper,
        )

        if not diagonal:
            self.add_mappers(axis=1, mapper=self.col_mapper)
        self.map_indices()

        row_indices = safe_concatenate_indices([obj.row for obj in self.groups], empty_ok)
        col_indices = safe_concatenate_indices([obj.col for obj in self.groups], empty_ok)

        if diagonal:
            x = int(self.row_mapper.index_array.max() + 1)
            dimensions = (x, x)
        else:
            dimensions = (
                int(self.row_mapper.index_array.max() + 1),
                int(self.col_mapper.index_array.max() + 1),
            )

        self.matrix = sparse.coo_matrix(
            (
                np.zeros(len(row_indices)),
                (row_indices, col_indices),
            ),
            dimensions,
            dtype=np.float64,
        ).tocsr()

        self.rebuild_matrix()

    def add_mappers(self, axis: int, mapper: ArrayMapper):
        for group in self.groups:
            group.add_mapper(axis=axis, mapper=mapper)

    def map_indices(self):
        for group in self.groups:
            group.map_indices(diagonal=self.diagonal)

    def iterate_indexers(self):
        for obj in self.packages:
            if hasattr(obj, "indexer"):
                next(obj.indexer)

    def rebuild_matrix(self):
        self.matrix.data *= 0
        for group in self.groups:
            row, col, data = group.calculate()
            if group.package.metadata["sum_inter_duplicates"]:
                self.matrix[row, col] += data
            else:
                self.matrix[row, col] = data

    def __next__(self):
        self.iterate_indexers()
        self.rebuild_matrix()

    def add_indexers(self, indexer_override: Any, seed_override: Union[int, None]):
        """Add indexers"""
        for package, resources in self.packages.items():
            if hasattr(package, "indexer"):
                for obj in resources:
                    obj.add_indexer(indexer=package.indexer)
            elif indexer_override is not None:
                package.indexer = indexer_override
                for obj in resources:
                    obj.add_indexer(indexer=package.indexer)
            elif package.metadata["combinatorial"]:
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
