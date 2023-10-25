from collections.abc import Mapping
from typing import Any, Callable, Optional, Sequence, Union

from bw_processing import Datapackage

from .array_mapper import ArrayMapper
from .mapped_matrix import MappedMatrix
from .indexers import RandomIndexer, SequentialIndexer, Indexer


class MappedMatrixDict(Mapping):
    def __init__(
        self,
        *,
        packages: dict[Union[tuple, str], Sequence[Datapackage]],
        matrix: str,
        row_mapper: ArrayMapper,
        col_mapper: ArrayMapper,
        use_vectors: bool = True,
        use_arrays: bool = True,
        use_distributions: bool = False,
        seed_override: Optional[int] = None,
        indexer_override: Any = None,
        diagonal: bool = False,
        transpose: bool = False,
        custom_filter: Optional[Callable] = None,
        empty_ok: bool = False,
        sequential: bool = False,
    ):
        """A thin wrapper around a dict of `MappedMatrix` objects. See its docstring
        for details on `custom_filter` and `indexer_override`.

        The matrices have the same dimensions, the same lookup dictionaries, and the
        same indexer.

        The number of possible configurations of resource groups and indexers is far
        higher than any generic class can handle. This class supports either
        sequential or random indexing, and the indexing is applied to **all resource
        groups and datapackages**. If you need finer-grained control, you can access
        set and access the individual resource group `indexer` attributes.

        Because the same indexer is used for all datapackages, individual `seed` values
        are ignored. Use `seed_override` to set a global RNG seed.

        Parameters
        ----------
        packages : dict[Union[tuple, str], Sequence[Datapackage]]
            A dictionary with identifiers as keys and a list of `bw_processing`
            datapackages as values.
        matrix : str
            The string identifying the matrix to be built.
        use_vectors : bool
            Flag to use vector data from datapackages
        use_arrays : bool
            Flag to use array data from datapackages
        use_distributions : bool
            Flag to use `stats_arrays` distribution data from datapackages
        row_mapper : ArrayMapper
            Used when matrices must align to an existing mapping.
        col_mapper :
            Used when matrices must align to an existing mapping.
        seed_override : int
            Overrides the RNG seed given in the datapackage, if any.
        indexer_override : Any
            Parameter for custom indexers. See above.
        diagonal : bool
            If `True`, only use the `row` indices to build a diagonal matrix.
        transpose : bool
            Transpose row and column indices. Happens before any processing, so filters
            and mappers should refer to the transposed dimensions.
        custom_filter : Callable
            Callable for function to filter data based on `indices` values. See above.
        empty_ok : bool
            If False, raise `AllArraysEmpty` if the matrix would be empty
        sequential : bool
            Use the **same sequential indexer** across all resource groups in all datapackages
        """
        self.matrix = matrix
        self.row_mapper = row_mapper
        self.col_mapper = col_mapper
        self.use_vectors = use_vectors
        self.use_arrays = use_arrays
        self.seed_override = seed_override
        self.diagonal = diagonal
        self.transpose = transpose
        self.custom_filter = custom_filter
        self.empty_ok = empty_ok
        self.global_indexer = self.get_global_indexer(
            indexer_override=indexer_override,
            sequential=sequential,
            seed_override=seed_override,
        )
        self.matrices = {
            obj: MappedMatrix(
                packages=packages,
                matrix=matrix,
                use_vectors=use_vectors,
                use_arrays=use_arrays,
                use_distributions=use_distributions,
                row_mapper=row_mapper,
                col_mapper=col_mapper,
                seed_override=seed_override,
                indexer_override=self.global_indexer,
                diagonal=diagonal,
                transpose=transpose,
                custom_filter=custom_filter,
                empty_ok=empty_ok,
            )
            for obj, packages in packages.items()
        }

    def __getitem__(self, key: Union[tuple, str]) -> MappedMatrix:
        return self.matrices[key]

    def __iter__(self):
        for obj in self.matrices:
            return self.matrices[obj]

    def __len__(self) -> int:
        return len(self.matrices)

    def __next__(self) -> None:
        next(self.global_indexer)
        for mm in self.matrices.values():
            mm.rebuild_matrix()

    def get_global_indexer(
        self, indexer_override: Any, sequential: bool, seed_override: Optional[int]
    ) -> Indexer:
        if indexer_override is not None:
            return indexer_override
        elif sequential:
            return SequentialIndexer()
        else:
            return RandomIndexer(seed=seed_override)
