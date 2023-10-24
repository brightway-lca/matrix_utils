from .mapped_matrix import MappedMatrix


class MappedMatrixList:
    """Class which handles a list of mapped matrices.

    The matrices have the same dimensions, the same lookup dictionaries, and the same indexer."""
    def __init__(
        self,
        *,
        packages: dict[str, Sequence[Datapackage]],
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
        """A thin wrapper around a list of `MappedMatrix` objects. See its docstring for details on `custom_filter` and `indexer_override`.

        Parameters
        ----------
        packages : list[Datapackage]
            A list of Datapackage objects.
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
            Transpose row and column indices. Happens before any processing, so filters and mappers should refer to the transposed dimensions.
        custom_filter : Callable
            Callable for function to filter data based on `indices` values. See above.
        empty_ok : bool
            If False, raise `AllArraysEmpty` if the matrix would be empty

        """
