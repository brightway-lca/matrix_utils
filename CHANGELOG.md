# Changelog

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
