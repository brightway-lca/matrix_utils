from collections.abc import Mapping
from typing import Any, Callable, Optional, Sequence, Union

from bw_processing import Datapackage

from .array_mapper import ArrayMapper
from .mapped_matrix import MappedMatrix


class MappedMatrixDict(Mapping):
    """Class which handles a list of mapped matrices.

    The matrices have the same dimensions, the same lookup dictionaries, and the same
    indexer.
    """

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
    ):
        """A thin wrapper around a list of `MappedMatrix` objects. See its docstring
        for details on `custom_filter` and `indexer_override`.

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

        """
        self.matrices = {
            tpl: MappedMatrix(
                packages=packages,
                matrix=matrix,
                use_vectors=use_vectors,
                use_arrays=use_arrays,
                use_distributions=use_distributions,
                row_mapper=row_mapper,
                col_mapper=col_mapper,
                seed_override=seed_override,
                indexer_override=None,
                diagonal=diagonal,
                transpose=transpose,
                custom_filter=custom_filter,
                empty_ok=empty_ok,
            )
            for tpl, packages in packages.items()
        }

    def __getitem__(self, key: Union[tuple, str]) -> MappedMatrix:
        return self.matrices[key]

    def __iter__(self):
        for tpl in self.matrices:
            return self.matrices[tpl]

    def __len__(self) -> int:
        return len(self.matrices)

    def __next__(self) -> None:
        pass
