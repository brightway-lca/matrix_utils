# matrix_utils

Library for building matrices from data packages from [bw_processing](https://github.com/brightway-lca/bw_processing0). Designed for use with the [Brightway life cycle assessment framework](https://brightway.dev/).

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [API](#api)
- [Contributing](#contributing)
- [Maintainers](#maintainers)
- [License](#license)

## Background

The [calculation library](https://github.com/brightway-lca/brightway2-calc) of the [Brightway LCA framework](https://brightway.dev/) has traditionally include matrix-building functionality. As the new capabilities in `bw_processing` have increased matrix-building complexity, this library is a refactoring to split matrix utilities from the LCA classes, which will remain in the calculation library.

`matrix_utils` supports all the features made available in `bw_processing`: static and dynamic resources, data package policies, vector and array resources. It also improves on the previous matrix building code by speeding up the mapping from data source ids to row and column ids.

### Backwards compatibility

This library presents a completely different API than the functions previously present in `bw2calc`.



## Install

Install using pip or conda (channel `cmutel`).

Depends on numpy, scipy, pandas, bw_processing, stats_arrays.

## Usage

### `MappedMatrix` class

The primary use case for `matrix_utils` is the `MappedMatrix` class:

```python

In [1]: from matrix_utils import MappedMatrix

In [2]: mm = MappedMatrix(packages=[some_datapackage], matrix="foo")

In [3]: mm.matrix
Out[3]:
<8x8 sparse matrix of type '<class 'numpy.float32'>'
    with 11 stored elements in Compressed Sparse Row format>
```

`MappedMatrix` takes the following arguments. Note that all arguments **must be** keyword arguments:

* `packages`: list, required. List of `bw_processing` data packages.
* `matrix`: str, required. Label of matrix to build. Used to filter data in `packages`, so must be idential to the package resource `matrix`.
* `use_vectors`: bool, default True. Include vector data resources when building matrices.
* `use_arrays`: bool, default True. Include array data resources when building matrices.
* `use_distributions`: bool, default False. Include probability distribution data resources when building matrices.
* `row_mapper`: `matrix_utils.ArrayMapper`, default None. Optional mapping class used to translate data source ids to matrix row ids. In LCA, one would reuse this mapping class to make sure the dimensions of multiple matrices align.
* `col_mapper`: `matrix_utils.ArrayMapper`, default None. Optional mapping class used to translate data source ids to matrix column ids. In LCA, one would reuse this mapping class to make sure the dimensions of multiple matrices align.
* `seed_override`: int, default None. Override the random seed given in the data package. Note that this is ignored if the data package is combinatorial.

`MappedMatrix` is iterable; calling `next()` will draw new samples from all included stochastic resources, and rebuild the matrix.

You may also find it useful to iterate through `MappedMatrix.groups`, which are instances of `ResourceGroup`, documented below.

### `ResourceGroup` class

A `bw_processing` data package is essentially a metadata file and a bag of data resources. These resources are *grouped*, for multiple resources are needed to build one matrix, or one component of one matrix. For example, one needs not on the data vector, but also the row and column indices (which are a separate file), to build a simple matrix. One could also have a `flip` vector, in another file, used to flip the signs of data elements before matrix insertion.

The `ResourceGroup` class provides a single interface to these data files and their metadata. `ResourceGroup` instances are created automatically by `MappedMatrix`, and available via `MappedMatrix.groups`. The [source code]() is pretty readable, and in general you probably don't need to worry about this low-level class, but the following could be useful:

* `ResourceGroup.data`: The Numpy data vector or array, or the data interface. This is the raw input data, duplicate elements are not aggregated (if applicable).
* `ResourceGroup.sample`: Numpy vector of the data inserted into the matrix, after aggregation (if applicable) and sign flipping.
* `ResourceGroup.indices`: The Numpy structured data array with **unmapped** indices (i.e. data source ids). Has `row` and `col` columns.
* `ResourceGroup.row`: Numpy vector of matrix row indices.
* `ResourceGroup.col`: Numpy vector of matrix col indices.
* `ResourceGroup.row_disaggregated`: Numpy vector of matrix row indices before summing duplicate entries. Only present is `sum_duplicates` is True.
* `ResourceGroup.row_disaggregated`: Numpy vector of matrix column indices before summing duplicate entries. Only present is `sum_duplicates` is True.
* `ResourceGroup.calculate(vector=None)`: Function to recalculate matrix row, column, and data vectors. Uses the current state of the indexers, but re-draws values from data iterators. If `vector` is given, use this instead of the given data source.
* `ResourceGroup.indexer`: The instance of the `Indexer` class applicable for this `ResourceGroup`. Only used for data arrays.
* `ResourceGroup.ncols`: The integer number of columns in a data array. Returns `None` is a data vector is present.

### Iterating through arrays



## Contributing

Your contribution is welcome! Please follow the [pull request workflow](https://guides.github.com/introduction/flow/), even for minor changes.

When contributing to this repository with a major change, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository.

Please note we have a [code of conduct](https://github.com/brightway-lca/bw_processing/blob/master/CODE_OF_CONDUCT.md), please follow it in all your interactions with the project.

### Documentation and coding standards

* [Black formatting](https://black.readthedocs.io/en/stable/)
* [Semantic versioning](http://semver.org/)

## Maintainers

* [Chris Mutel](https://github.com/cmutel/)

## License

[BSD-3-Clause](https://github.com/brightway-lca/matrix_utils/blob/main/LICENSE). Copyright 2020 Chris Mutel.
