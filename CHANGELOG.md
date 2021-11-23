# Changelog

## [0.2] - DEV

### Changed attributes in `ResourceGroup`

Rewrote basic functionality of `ResourceGroup` to make it clear what data was masked and was wasn't. Removed `raw_indices`, `indices`, `raw_flip`, `row`, `col`, and `data`. We now use a uniform naming convention for `data`, `row`, and `col`:

* `X_original`: The data as it is present in the datapackage. The only except are indices, which might be transposed (but not masked).
* `X_masked`: The data after the custom filter and mapping mask have been applied.
* `X_matrix`: Row and column indices (but not data) for insertion into the matrix. These indices are after aggregation within the resource group (if any).

`flip` is always masked. `current_data` is now `data_current` (to be consistent with the naming convention), and is both masked and flipped, but not aggregated. `indices` are removed, use `get_indices_data()`.

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
