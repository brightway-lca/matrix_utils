# Changelog

### [0.4.1] - 2023-12-18

* Allow passing `MappedMatrix` subclass to `MappedMatrixDict`

## [0.4] - 2023-12-07

* Add `SparseMatrixDict`
* Defined matrix multiplication for `MappedMatrixDict`

## [0.3] - 2023-10-25

* Add `MappedMatrixDict` class for `MultiLCA`

### [0.2.5] - 2022-05-21

* Add functions to reset indexers directly and on `MappedMatrix`

### [0.2.4] - 2022-05-15

* Avoid `StopIteration` errors on empty combinatorial filtered datapackages

### [0.2.3] - 2022-02-02

* Fix an error where the previous attribute `current_data` was not consistently changed to `data_current`

### [0.2.2] - 2021-11-26

* Fix an error raised when a `ResourceGroup` was empty after masking

### [0.2.1] - 2021-11-26

* Add `input_provenance` function on showing from which datapackage and resource group data comes from

## [0.2] - 2021-11-25

### Added attributes in `MappedMatrix` for sensitivity analysis

Added the following attributes to `MappedMatrix`:

* `input_data_vector`: The stacked data vector from all resource groups.
* `input_row_col_indices`: The stacked indices structured array (with columns `row` and `col`) from all resource groups.
* `input_indexer_vector`: A vector of values from the resource group indexers. This array flattens value from combinatorial indexers, be careful that you understand the values.
* `input_uncertainties`: The stacked distributions structured array (dtype `bw_processing.UNCERTAINTY_DTYPE`) from all resource groups. See the documentation for how this works with arrays and interfaces.

Note that these values are before any aggregation policies are applied, i.e. values to be placed in the same matrix cell are not summed or overwritten.

### Changed attributes in `ResourceGroup`

Rewrote basic functionality of `ResourceGroup` to make it clear what data was masked and was wasn't. Removed `raw_indices`, `indices`, `raw_flip`, `row`, `col`, and `data`. We now use a uniform naming convention for `data`, `row`, and `col`:

* `data_original`: The data as it is present in the datapackage, before masking.
* `data_current`: The data sample used (before aggregation) to build the matrix. It is both masked and flipped.
* `row|col_mapped`: Mapped row and column indices. Has the same length as the datapackage resource, but uses `-1` for values which weren't mapped.
* `row|col_masked`: The data after the custom filter and mapping mask have been applied.
* `row|col_matrix`: Row and column indices (but not data) for insertion into the matrix. These indices are after aggregation within the resource group (if any).

`flip` is always masked. `current_data` is now `data_current` (to be consistent with the naming convention), and is both masked and flipped, but not aggregated.

`.indices` are removed, use `.get_indices_data()`.

### [0.1.4] - 2021-10-20

* Add `transpose` flag when creating matrices

### [0.1.3] - 2021-06-16

* Allow array data with `use_distrbutions` (instead of raising an error)

### [0.1.2] - 2021-06-15

* Fix bug in boolean logic order of operations in `has_relevant_data`

### [0.1.1] - 2021-06-02

* Add `custom_filter` argument to filter out some package data based on indices values.
* Add `indexer_override` argument for custom indexers.
* Respect existing indexers on datapackages.
* `filter_groups_for_packages` as a separate utility function instead of in `DataPackage` class.
* Allow creation of `(0,0)` matrices if no data arrays are present.

## [0.1] - 2021-05-19

First complete public release.
