from .resource_group import ResourceGroup
from .array_mapper import ArrayMapper
from bw_processing import load_datapackage
from typing import Union
import numpy as np


class MappedMatrix:
    """A scipy sparse matrix handler which takes in ``bw_processing`` data packages. Row and column values are mapped to indices, and a matrix is constructed."""

    def __init__(
        self,
        *,
        packages: list,
        matrix_label: str,
        use_static: bool = True,
        use_substitutions: bool = True,
        use_distributions: bool = False,
        row_mapper: Union[ArrayMapper, None] = None,
        col_mapper: Union[ArrayMapper, None] = None,
        seed: Union[int, None] = None
    ):
        self.matrix_label = matrix_label
        # mmap mode?
        self.packages = [
            load_datapackage(obj).filter_by_attribute("matrix_label", matrix_label)
            for obj in packages
        ]
        self.groups = [
            ResourceGroup(package=package, group_label=group_label)
            for obj in self.packages
            for group_label, package in obj.groups().items()
        ]

        self.row_mapper = row_mapper
        if self.row_mapper is None:
            self.row_mapper = ArrayMapper(
                array=np.hstack([obj.unique_row_indices() for obj in self.groups]),
                ensure_uniqueness=False,
            )

        self.col_mapper = col_mapper
        if self.col_mapper is None:
            self.col_mapper = ArrayMapper(
                array=np.hstack([obj.unique_col_indices() for obj in self.groups]),
                ensure_uniqueness=False,
            )
