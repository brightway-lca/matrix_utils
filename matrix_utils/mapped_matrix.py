from typing import Any, Callable, Optional, Sequence, List

import numpy as np
from bw_processing import INDICES_DTYPE, UNCERTAINTY_DTYPE, Datapackage
from scipy import sparse

from .array_mapper import ArrayMapper
from .errors import EmptyInterface
from .indexers import CombinatorialIndexer, RandomIndexer, SequentialIndexer
from .resource_group import ResourceGroup
from .utils import filter_groups_for_packages, safe_concatenate_indices


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
        * transpose: Transpose row and column indices. Happens before any processing, so filters and mappers should refer to the transposed dimensions.
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
        row_mapper: Optional[ArrayMapper] = None,
        col_mapper: Optional[ArrayMapper] = None,
        seed_override: Optional[int] = None,
        indexer_override: Any = None,
        diagonal: bool = False,
        transpose: bool = False,
        custom_filter: Optional[Callable] = None,
        empty_ok: bool = False,
    ):
        self.seed_override = seed_override
        self.diagonal = diagonal
        self.matrix_label = matrix
        self.use_distributions = use_distributions
        self.use_vectors = use_vectors
        self.use_arrays = use_arrays
        self.packages = {
            package: [
                ResourceGroup(
                    package=filtered_package,
                    group_label=group_label,
                    use_distributions=use_distributions,
                    seed_override=seed_override,
                    custom_filter=custom_filter,
                    transpose=transpose,
                )
                for group_label, filtered_package in lst
            ]
            for package, lst in filter_groups_for_packages(
                packages, matrix, use_vectors, use_arrays, use_distributions
            ).items()
        }

        for package in self.packages:
            if package.dehydrated_interfaces():
                raise EmptyInterface(
                    "Dehydrated interfaces {} in package {} need to be rehydrated to be used in matrix calculations".format(
                        package.dehydrated_interfaces(), package
                    )
                )

        self.groups = tuple([obj for lst in self.packages.values() for obj in lst])
        self.add_indexers(indexer_override, seed_override)

        self.row_mapper = row_mapper or ArrayMapper(
            array=safe_concatenate_indices(
                [obj.unique_row_indices_for_mapping() for obj in self.groups], empty_ok
            ),
            empty_ok=empty_ok,
        )
        if not diagonal:
            self.col_mapper = col_mapper or ArrayMapper(
                array=safe_concatenate_indices(
                    [obj.unique_col_indices_for_mapping() for obj in self.groups],
                    empty_ok,
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

        row_indices = safe_concatenate_indices(
            [obj.row_matrix for obj in self.groups], empty_ok
        )
        col_indices = safe_concatenate_indices(
            [obj.col_matrix for obj in self.groups], empty_ok
        )

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
            # Avoid ``StopIteration`` errors if packaged is filtered to emptiness
            if hasattr(obj, "indexer") and self.packages[obj]:
                next(obj.indexer)

    def reset_indexers(self, rebuild=False):
        for obj in self.packages:
            if hasattr(obj, "indexer"):
                obj.indexer.reset()
        if rebuild:
            self.rebuild_matrix()

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

    def add_indexers(self, indexer_override: Any, seed_override: Optional[int]):
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

    def input_data_vector(self) -> np.ndarray:
        return np.hstack([group.data_current for group in self.groups])

    def input_row_col_indices(self) -> np.ndarray:
        rows, cols = [], []

        for group in self.groups:
            rows.append(group.row_masked)
            cols.append(group.col_masked)

        rows, cols = np.hstack(rows), np.hstack(cols)

        array = np.empty(shape=(len(rows),), dtype=INDICES_DTYPE)
        array["row"] = rows
        array["col"] = cols
        return array

    def input_provenance(self) -> List[tuple]:
        """Describe where the data in the other ``input_X`` comes from. Returns a list of ``(datapackage, group_label, (start_index, end_index))`` tuples.

        Note that the ``end_index`` is exclusive, following the Python slicing convention, i.e. ``(7, 9)`` means start from the 8th element (indices start from 0), and go up to but don't include the 10th element (i.e. (7, 9) has two elements)."""
        position, result = 0, []

        for package, groups in self.packages.items():
            for group in groups:
                num_elements = len(group.data_current)
                # Minus one because we include the first element as element 0
                result.append((package, group.label, (position, position + num_elements)))
                # Plus one because start at the next value
                position += num_elements
        return result

    def input_indexer_vector(self) -> np.ndarray:
        index_values = []

        for package in self.packages:
            value = package.indexer.index
            if isinstance(value, (int, np.int32, np.int64)):
                index_values.append(int(value))
            elif isinstance(value, tuple):
                index_values.extend(list(value))
            else:
                raise ValueError(
                    f"Can't understand indexer value {value} in package {package}"
                )
        return np.array(index_values)

    def _construct_distributions_array(self, given, uncertainty_type=0) -> np.ndarray:
        FIELDS = ["scale", "shape", "minimum", "maximum"]

        array = np.zeros(len(given), dtype=UNCERTAINTY_DTYPE)
        for field in FIELDS:
            array[field] = np.NaN
        array["loc"] = given
        array["uncertainty_type"] = uncertainty_type
        return array

    def input_uncertainties(self, number_samples: Optional[int] = None) -> np.ndarray:
        """Return the stacked uncertainty arrays of all resources groups.

        Note that this data is masked with both the custom filter (if present) and the mapping mask!

        If the resource group has a distributions array, then this is returned. Otherwise, if the data is static, a distributions array with uncertainty type 0 (undefined uncertainty) is constructed. If the data is an array, an estimate of the mean and standard deviation are given in the ``loc`` and ``scale`` columns. This estimate uses ``number_samples`` columns, or all columns if ``number_samples`` is ``None``.

        If the data comes from an interface, a distributions array with uncertainty type 0 will be created. Regardless if whether it is a vector or an array interface, the current data vector is used, and no estimate of uncertainty is made. Therefore, this data will never consume new data from an interface.

        Raises a ``TypeError`` if distributions arrays are present but don't follow the dtype of ``bw_processing.UNCERTAINTY_DTYPE``.

        As both population samples (arrays) and interfaces don't fit into the traditional ``stat_arrays`` framework, we mark these with custom ``uncertainty_types``:

        * ``98`` for arrays
        * ``99`` for interfaces

        """
        arrays = []

        for group in self.groups:
            if group.has_distributions and self.use_distributions:
                if group.data_original.dtype != UNCERTAINTY_DTYPE:
                    raise TypeError(
                        "Distributions datatype should be `bw_processing.UNCERTAINTY_DTYPE`, but was {}".format(
                            group.data_original.dtype
                        )
                    )
                arrays.append(group.apply_masks(group.data_original))
            elif group.is_array() and not group.is_interface():
                data = group.data_original
                if number_samples is not None:
                    data = data[:, :number_samples]

                array = self._construct_distributions_array(
                    group.data_current, uncertainty_type=98
                )
                array["loc"] = np.mean(data, axis=1)
                array["loc"][group.flip] *= -1
                array["scale"] = np.std(data, axis=1)
                arrays.append(array)
            elif group.is_interface():
                arrays.append(
                    self._construct_distributions_array(
                        group.data_current, uncertainty_type=99
                    )
                )
            else:
                arrays.append(self._construct_distributions_array(group.data_current))

        return np.hstack(arrays)
