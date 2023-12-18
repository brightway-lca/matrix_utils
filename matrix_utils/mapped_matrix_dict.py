from collections.abc import Mapping
from typing import Any, Callable, Optional, Sequence, Type, Union

from bw_processing import Datapackage
from scipy import sparse

from .array_mapper import ArrayMapper
from .indexers import Indexer, RandomIndexer, SequentialIndexer
from .mapped_matrix import MappedMatrix


class SparseMatrixDict(dict):
    """Instantiate with `SparseMatrixDict({"label": sparse_matrix})`"""

    def __matmul__(self, other):
        """Define logic for `@` matrix multiplication operator.

        Note that the sparse matrix dict must come first, i.e. `SparseMatrixDict @ other`.
        """
        if isinstance(other, SparseMatrixDict):
            return SparseMatrixDict(
                {(a, b): c @ d for a, c in self.items() for b, d in other.items()}
            )
        if isinstance(other, MappedMatrixDict):
            return SparseMatrixDict(
                {(a, b): c @ d.matrix for a, c in self.items() for b, d in other.items()}
            )
        elif sparse.base.issparse(other):
            return SparseMatrixDict({a: b @ other for a, b in self.items()})
        else:
            raise TypeError(f"Can't understand matrix multiplication for type {type(other)}")


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
        matrix_class: Type[MappedMatrix] = MappedMatrix,
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

        The `empty_ok` flag applies to **all matrices** - if any of the matrices have
        a valid data value no error will be raised. In practice this flag should have
        no effect for `MappedMatrixDict` unless the input data is very broken.

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
            Use the **same sequential indexer** across all resource groups in all
            datapackages
        matrix_class : MappedMatrix
            `MappedMatrix` class to use. Can be a subclass of `MappedMatrix`.
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

        if not isinstance(packages, Mapping):
            raise ValueError("`packages` must be a dictionary")

        self.matrices = {
            key: matrix_class(
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
            for key, packages in packages.items()
        }

    def __getitem__(self, key: Any) -> MappedMatrix:
        return self.matrices[key]

    def __iter__(self):
        yield from self.matrices

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

    def __matmul__(self, other):
        """Define logic for `@` matrix multiplication operator.

        A `MappedMatrixDict` can only be multiplied by a sparse matrix; no other type is supported.

        Note that the mapped matrix dict must come first, i.e. `MappedMatrixDict @ other`.
        """
        if sparse.base.issparse(other):
            return SparseMatrixDict(
                [(key, value.matrix @ other) for key, value in self.matrices.items()]
            )
        else:
            raise TypeError(f"Can't understand matrix multiplication for type {type(other)}")
