from typing import Any, Callable, Optional, Union

import numpy as np
from bw_processing import DatapackageBase
from stats_arrays import MCRandomNumberGenerator

from matrix_utils.array_mapper import ArrayMapper
from matrix_utils.indexers import Indexer, Proxy


class FakeRNG:
    def __init__(self, array):
        self.array = array

    def __next__(self):
        return self.array


def mask_array(array, mask=None):
    if mask is not None:
        return array[mask]
    else:
        return array


class ResourceGroup:
    """A class that handles a resource group - a collection of files in data package which define one matrix. A resource group can contain the following:

    * data array or interface (required)
    * indices structured array (required)
    * flip array (optional)
    * csv metadata (optional)
    * json metadata (optional)

    After instantiation, the ``MappedMatrix`` class will add an indexer (an instance of ``matrix_utils.indexers.Indexer``), using either ``add_indexer`` or ``add_combinatorial_indexer``. It will also add one or two array mappers, using ``add_mapper``. Only one mapper is needed if the matrix is diagonal.

    One easy source of confusion is the difference between ``.row_mapped`` and ``.row``. Both of these are one-dimensional vectors which have matrix row indices (i.e. they have already been mapped). ``.row_mapped`` are the values as given in the data package, whereas ``.row`` are the values actually used in matrix data insertion (and the same for column indices).There are two possible modifications that can be applied from ``.row_mapped`` to ``.row``; first, we can have internal aggregation, which will shrink the size of the row or column indices, as duplicate elements are eliminated. Second, if the array mapper was already instantiated, we might need to delete some elements from the row or column indices, as these values are not used in this calculation. For example, in life cycle assessment, LCIA method implementations often contain characterization factors for flows not present in the biosphere (as they were not used by an of the activities). In this case, we would need to eliminate these factors, as our characterization matrix must match exactly to the biosphere matrix already built.

    Here is an example for row indices:

    .. code-block:: python

        row_input_indices = [0, 17, 99, 42, 17]
        row_mapped = [0, 1, -1, 2, 1]
        after_aggregation = [0, 1, -1, 2]  # -1 is missing data point
        after_masking = [0, 1, 2]
        row = [0, 1, 2]

    Any data coming into this class, with through instantiation or via method calls such as ``.calculate``, should follow the length and order of the original datapackage data (use ``get_indices_data()`` to get the original indices.

    The current data, as entered into the matrix, is given by ``.current_data``.

    """  # NOQA: E501

    def __init__(
        self,
        *,
        package: DatapackageBase,
        group_label: str,
        use_distributions: bool = False,
        seed_override: Union[int, None] = None,
        custom_filter: Union[Callable, None] = None,
        transpose: bool = False,
    ):
        self.label = group_label
        self.package = package
        self.use_distributions = use_distributions
        self.custom_filter = custom_filter
        self.transpose = transpose
        self.vector = self.is_vector()
        self.seed = (
            seed_override if seed_override is not None else self.package.metadata.get("seed")
        )

        if custom_filter is not None:
            self.custom_filter_mask = custom_filter(self.get_indices_data())
        else:
            self.custom_filter_mask = None

        if self.use_distributions and self.vector:
            if self.has_distributions:
                self.rng = MCRandomNumberGenerator(params=self.data_original, seed=self.seed)
            else:
                self.rng = FakeRNG(self.data_original)

        self.aggregate = self.package.metadata["sum_intra_duplicates"]
        self.empty = self.get_indices_data().shape == (0,)

    def __str__(self):
        return """ResourceGroup {}\n\tVector: {}\n\tDistributions: {}\n\tTranspose: {}\n\tSeed: {}
\tCustom filter: {}""".format(
            self.label,
            self.vector,
            self.use_distributions,
            self.transpose,
            self.seed,
            bool(self.custom_filter),
        )

    @property
    def identifier(self) -> Optional[str]:
        """Shortcut to get a single `identifier` attribute for the group resources"""
        ids = {
            group["identifier"]
            for group in self.package.metadata["resources"]
            if group.get("identifier")
        }
        if not ids or len(ids) > 1:
            return None
        else:
            return ids.pop()

    @property
    def has_distributions(self):
        try:
            self.get_resource_by_suffix("distributions")
            return True
        except KeyError:
            return False

    @property
    def has_flip(self) -> bool:
        try:
            self.get_resource_by_suffix("flip")
            return True
        except KeyError:
            return False

    @property
    def has_rescale(self) -> bool:
        try:
            self.get_resource_by_suffix("rescale")
            return True
        except KeyError:
            return False

    @property
    def has_reference(self) -> bool:
        try:
            self.get_resource_by_suffix("reference")
            return True
        except KeyError:
            return False

    @property
    def has_params(self) -> bool:
        try:
            self.get_resource_by_suffix("params")
            return True
        except KeyError:
            return False

    @property
    def has_param_labels(self) -> bool:
        try:
            self.get_resource_by_suffix("param_labels")
            return True
        except KeyError:
            return False

    @property
    def params_current(self) -> np.ndarray:
        """Current parameter values.

        For vectors: returns the full 1-D params array (parameters are fixed
        and do not vary by indexer position).
        For arrays: returns the column at the current indexer position,
        mirroring how ``calculate()`` picks the active data column.

        Raises ``KeyError`` if no params array exists for this group.
        """
        data = self.get_resource_by_suffix("params")
        if self.vector:
            return data
        return data[:, self.indexer.index % data.shape[1]]

    @property
    def param_labels(self) -> dict:
        """Raw param-labels dict. Always has ``"values"``;
        also has ``"schema"`` when a label schema was provided at write time.

        Raises ``KeyError`` if no param_labels resource exists.
        """
        return self.get_resource_by_suffix("param_labels")

    @property
    def n_elements_dropped(self) -> int:
        """Number of original datapackage elements dropped by the custom filter and mapping mask."""
        return len(self.get_indices_data()) - len(self.row_masked)

    @property
    def data_original(self):
        if self.use_distributions and self.has_distributions:
            return self.get_resource_by_suffix("distributions")
        else:
            return self.get_resource_by_suffix("data")

    @property
    def flip(self):
        """The flip array, with all masks applied (if necessary).

        After ``map_indices``, this is the copy taken at mapping time, which is what
        ``calculate`` uses. Later changes to the datapackage array are ignored."""
        flip = self.__dict__.get("_flip")
        if flip is None:
            return self.apply_masks(self.get_resource_by_suffix("flip"))
        return flip

    @property
    def rescale(self):
        """The rescale array, with all masks applied (if necessary).

        After ``map_indices``, this is the copy taken at mapping time, which is what
        ``calculate`` uses. Later changes to the datapackage array are ignored."""
        rescale = self.__dict__.get("_rescale")
        if rescale is None:
            return self.apply_masks(self.get_resource_by_suffix("rescale"))
        return rescale

    @property
    def rescale_current(self) -> Optional[np.ndarray]:
        """Current rescale values with masks applied, or ``None`` if no rescale array exists."""
        if self.has_rescale:
            return self.rescale
        return None

    @property
    def reference(self):
        """The reference (production) exchange boolean array, with masks applied (if necessary)."""
        return self.apply_masks(self.get_resource_by_suffix("reference"))

    @property
    def reference_current(self) -> Optional[np.ndarray]:
        """Current reference flags with masks applied, or ``None`` if no reference array exists."""
        if self.has_reference:
            return self.reference
        return None

    def get_indices_data(self):
        """The source data for the indices array."""
        indices = self.get_resource_by_suffix("indices")
        if self.transpose:
            indices = indices.astype(
                [("col", indices.dtype[0]), ("row", indices.dtype[0])], copy=False
            )
        return indices

    def get_resource_by_suffix(self, suffix: str) -> Any:
        return self.package.get_resource(self.label + "." + suffix)[0]

    def is_vector(self) -> bool:
        """Determine if this is a vector or array resource"""
        metadata = self.package.get_resource(self.label + ".data")[1]
        return metadata.get("category") == "vector"

    def is_array(self) -> bool:
        """Determine if this is a vector or array resource"""
        return not self.is_vector()

    def is_interface(self) -> bool:
        """Determine if data is an interface"""
        return self.package.get_resource(self.label + ".data")[1]["profile"] == "interface"

    @property
    def ncols(self):
        if self.vector:
            return None
        else:
            return self.data_original.shape[1]

    def add_mapper(self, axis: int, mapper: ArrayMapper):
        if axis == 0:
            self.row_mapper = mapper
        else:
            self.col_mapper = mapper

    def build_mask(self, row, col):
        """Build boolean array mask where ``False`` means that a data element is not present, and
        should be ignored. See discussion above."""
        mask = (row != -1) * (col != -1)
        if (~mask).sum():
            return mask
        else:
            return None

    def map_indices(self, *, diagonal=False):
        self._mask_index = self._flip = self._rescale = self._multiplier = None

        if self.empty:
            self.row_matrix = self.row_masked = np.array([], dtype=np.int64)
            self.col_matrix = self.col_masked = np.array([], dtype=np.int64)
            return

        indices = self.get_indices_data()

        self.row_mapped = self.row_mapper.map_array(indices["row"])
        if diagonal:
            self.col_mapped = self.row_mapped
        else:
            self.col_mapped = self.col_mapper.map_array(indices["col"])

        self.unmapped_mask = self.build_mask(
            mask_array(self.row_mapped, self.custom_filter_mask),
            mask_array(self.col_mapped, self.custom_filter_mask),
        )
        self._n_elements = len(indices)
        self._mask_index = self._combined_mask_index()

        self.row_masked = self.apply_masks(self.row_mapped)
        self.col_masked = self.apply_masks(self.col_mapped)

        if self.row_masked.shape == (0,):
            self.row_matrix = self.row_masked
            self.col_matrix = self.col_masked
            self.empty = True
            return

        # Flip and rescale are copied here, so that ``calculate``, the ``flip`` and
        # ``rescale`` properties, and the ``MappedMatrix.input_*`` vectors always agree.
        if self.has_flip:
            self._flip = np.array(self.apply_masks(self.get_resource_by_suffix("flip")))
        if self.has_rescale:
            self._rescale = np.array(self.apply_masks(self.get_resource_by_suffix("rescale")))
        self._multiplier = self._build_multiplier()

        if self.aggregate:
            # Sort unique (row, col) pairs by the linear index ``row * count + col``, i.e.
            # by row and then column. Storing the inverse lets ``calculate`` sum
            # duplicates with a single ``bincount``.
            count = max(self.row_masked.max(), self.col_masked.max()) + 1
            linear = self.row_masked.astype(np.int64) * count + self.col_masked
            unique, inverse = np.unique(linear, return_inverse=True)
            self._aggregation_inverse = inverse.reshape(-1)
            self.row_matrix = (unique // count).astype(self.row_masked.dtype)
            self.col_matrix = (unique % count).astype(self.col_masked.dtype)
        else:
            self.row_matrix = self.row_masked
            self.col_matrix = self.col_masked

    def _combined_mask_index(self) -> Optional[np.ndarray]:
        """Combine ``custom_filter_mask`` and ``unmapped_mask`` into one integer index into the
        original data, or ``None`` if no element is masked."""
        if self.custom_filter_mask is None and self.unmapped_mask is None:
            return None
        # Indexing ``arange`` gives the same selection as indexing the data directly,
        # for any mask type which Numpy accepts as an index
        index = np.arange(self._n_elements)
        if self.custom_filter_mask is not None:
            index = index[self.custom_filter_mask]
        if self.unmapped_mask is not None:
            index = index[self.unmapped_mask]
        return index

    def _build_multiplier(self) -> Optional[np.ndarray]:
        """Combine the flip and rescale arrays (masked) into one multiplier, or ``None`` if
        neither exists. Flipping is exact in floating point, so ``data * (sign * rescale)``
        equals flipping and then rescaling."""
        multiplier = None
        if self._flip is not None:
            multiplier = np.where(self._flip, -1, 1).astype(np.int8)
        if self._rescale is not None:
            multiplier = self._rescale if multiplier is None else multiplier * self._rescale
        return multiplier

    def unique_row_indices_for_mapping(self):
        """Return array of unique indices that respect aggregation policy"""
        data = mask_array(self.get_indices_data()["row"], self.custom_filter_mask)
        # Structured-array field views can be non-contiguous; materialize contiguous memory first
        # to avoid pathological performance in `np.unique`.
        return np.unique(np.ascontiguousarray(data))

    def unique_col_indices_for_mapping(self):
        """Return array of unique indices that respect aggregation policy"""
        data = mask_array(self.get_indices_data()["col"], self.custom_filter_mask)
        # Structured-array field views can be non-contiguous; materialize contiguous memory first
        # to avoid pathological performance in `np.unique`.
        return np.unique(np.ascontiguousarray(data))

    def row_indices_for_mapping(self):
        """Return raw row indices for mapping, with custom filter applied if present."""
        data = mask_array(self.get_indices_data()["row"], self.custom_filter_mask)
        # Force contiguous memory to improve downstream sorting/uniquing performance.
        return np.ascontiguousarray(data)

    def col_indices_for_mapping(self):
        """Return raw column indices for mapping, with custom filter applied if present."""
        data = mask_array(self.get_indices_data()["col"], self.custom_filter_mask)
        # Force contiguous memory to improve downstream sorting/uniquing performance.
        return np.ascontiguousarray(data)

    def add_indexer(self, indexer: Indexer):
        self.indexer = indexer

    def add_combinatorial_indexer(self, indexer: Indexer, offset: int):
        self.indexer = Proxy(indexer, offset)

    def apply_masks(self, array):
        """Apply both ``custom_filter_mask`` (if present) and ``unmapped_mask``.

        Uses the combined index built in ``map_indices``, so can't be called before it."""
        index = self._mask_index
        return array if index is None else array[index]

    def calculate(self, vector: np.ndarray = None):
        """Generate row and column indices and a data vector. If ``.data`` is an iterator, draw the
        next value. If ``.data`` is an array, use the column given by ``.indexer``.

        ``vector`` is an optional input that overrides the data. It must be in the same order and
        have the same length as the data package indices (before possible aggregation and masking);
        see discussion above.

        """
        if self.empty:
            self.data_current = np.array([])
            return self.row_matrix, self.col_matrix, self.data_current

        if vector is not None:
            data = vector
        elif self.vector:
            if self.use_distributions and self.has_distributions:
                data = next(self.rng)
            else:
                try:
                    data = next(self.data_original)
                except TypeError:
                    data = self.data_original
        else:
            data = self.data_original[:, self.indexer.index % self.ncols]

        # `data` is now in the original state before **any** masking.
        if len(data) != self._n_elements:
            raise ValueError(
                f"Data has {len(data)} elements, but the indices of resource group "
                f"{self.label} have {self._n_elements}"
            )

        # Apply both custom filter mask and mapping mask. Fancy indexing returns a copy;
        # without masks, copy explicitly to avoid modifying the original data.
        if self._mask_index is not None:
            data = data[self._mask_index]
        else:
            data = data.copy()

        # Flip and rescale, combined into one multiplier in ``map_indices``
        if self._multiplier is not None:
            data *= self._multiplier

        # Second copy because we want to store the data before aggregation, and the
        # returned array shouldn't share memory with ``data_current``
        self.data_current = data.copy()

        if self.aggregate:
            # ``bincount`` always sums in float64; return the input dtype, as before
            aggregated = np.bincount(
                self._aggregation_inverse, weights=data, minlength=len(self.row_matrix)
            ).astype(data.dtype, copy=False)
            return self.row_matrix, self.col_matrix, aggregated
        else:
            return self.row_matrix, self.col_matrix, data
