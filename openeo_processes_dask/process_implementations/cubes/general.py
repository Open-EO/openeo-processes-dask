import copy
from typing import Optional, Union

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from openeo_pg_parser_networkx.pg_schema import *

from openeo_processes_dask.process_implementations.data_model import RasterCube
from openeo_processes_dask.process_implementations.exceptions import (
    DimensionLabelCountMismatch,
    DimensionNotAvailable,
)

__all__ = [
    "create_data_cube",
    "drop_dimension",
    "dimension_labels",
    "add_dimension",
    "rename_dimension",
    "rename_labels",
    "trim_cube",
]


def drop_dimension(data: RasterCube, name: str) -> RasterCube:
    if name not in data.dims:
        raise DimensionNotAvailable(
            f"Provided dimension ({name}) not found in data.dims: {data.dims}"
        )
    if len(data[name]) > 1:
        raise DimensionLabelCountMismatch(
            f"The number of dimension labels exceeds one, which requires a reducer. Dimension ({name}) has {len(data[name])} labels."
        )
    return data.drop_vars(name).squeeze(name)


def create_data_cube() -> RasterCube:
    return xr.DataArray()


def trim_cube(data) -> RasterCube:
    for dim in data.dims:
        if (
            dim in data.openeo.temporal_dims
            or dim in data.openeo.band_dims
            or dim in data.openeo.other_dims
        ):
            values = data[dim].values
            other_dims = [d for d in data.dims if d != dim]
            available_data = values[(np.isnan(data)).all(dim=other_dims) == 0]
            if len(available_data) == 0:
                raise ValueError(f"Data contains NaN values only. ")
            data = data.sel({dim: available_data})

    return data


def dimension_labels(data: RasterCube, dimension: str) -> ArrayLike:
    if dimension not in data.dims:
        raise DimensionNotAvailable(
            f"Provided dimension ({dimension}) not found in data.dims: {data.dims}"
        )

    coords = data.coords[dimension]
    if np.issubdtype(coords.dtype, np.datetime64):
        return np.datetime_as_string(coords, timezone="UTC")
    else:
        return np.array(data.coords[dimension])


def add_dimension(
    data: RasterCube, name: str, label: str, type: Optional[str] = "other"
):
    """
    Parameters
    ----------
    data : xr.DataArray
       A data cube to add the dimension to.
    name : str
       Name for the dimension.
    labels : number, str
       A dimension label.
    type : str, optional
       The type of dimension, defaults to other.
    Returns
    -------
    xr.DataArray :
       The data cube with a newly added dimension. The new dimension has exactly one dimension label.
       All other dimensions remain unchanged.
    """
    if name in data.dims:
        raise Exception(
            f"DimensionExists - A dimension with the specified name already exists. The existing dimensions are: {data.dims}"
        )
    data_e = data.assign_coords(**{name: label})
    data_e = data_e.expand_dims(name)
    # Register dimension in the openeo accessor
    data_e.openeo.add_dim_type(name=name, type=type)
    return data_e


def rename_dimension(
    data: RasterCube,
    source: str,
    target: str,
):
    """
    Parameters
    ----------
    data : xr.DataArray
       A data cube.
    source : str
       The current name of the dimension.
       Fails with a DimensionNotAvailable exception if the specified dimension does not exist.
    labels : number, str
       A new Name for the dimension.
       Fails with a DimensionExists exception if a dimension with the specified name exists.
    Returns
    -------
    xr.DataArray :
       A data cube with the same dimensions,
       but the name of one of the dimensions changes.
       The old name can not be referred to any longer.
       The dimension properties (name, type, labels, reference system and resolution)
       remain unchanged.
    """
    if source not in data.dims:
        raise DimensionNotAvailable(
            f"Provided dimension ({source}) not found in data.dims: {data.dims}"
        )
    if target in data.dims:
        raise Exception(
            f"DimensionExists - A dimension with the specified name already exists. The existing dimensions are: {data.dims}"
        )
    # Register dimension in the openeo accessor
    if source in data.openeo.spatial_dims:
        dim_type = "spatial"
    elif source in data.openeo.temporal_dims:
        dim_type = "temporal"
    elif source in data.openeo.band_dims:
        dim_type = "bands"
    else:
        dim_type = "other"
    data = data.rename({source: target})
    data.openeo.add_dim_type(name=target, type=dim_type)
    return data


def rename_labels(
    data: RasterCube,
    dimension: str,
    target: list[Union[str, float]],
    source: Optional[list[Union[str, float]]] = [],
):
    data_rename = copy.deepcopy(data)
    if dimension not in data_rename.dims:
        raise DimensionNotAvailable(
            f"Provided dimension ({dimension}) not found in data.dims: {data_rename.dims}"
        )
    if source:
        if len(source) != len(target):
            raise Exception(
                f"LabelMismatch - The number of labels in the parameters `source` and `target` don't match."
            )

    source_labels = data_rename[dimension].values
    if isinstance(source_labels, np.ndarray):
        source_labels = source_labels.tolist()
    if isinstance(target, np.ndarray):
        target = target.tolist()

    target_values = []

    for label in source_labels:
        if label in target:
            raise Exception(f"LabelExists - A label with the specified name exists.")
        if source:
            if label in source:
                target_values.append(target[source.index(label)])
            else:
                target_values.append(label)

    if not source:
        if len(source_labels) == len(target):
            data_rename[dimension] = target
        elif len(target) < len(source_labels):
            if 0 in source_labels:
                target_values = target + source_labels[len(target) :]
                data_rename[dimension] = target_values
            else:
                raise Exception(
                    f"LabelsNotEnumerated - The dimension labels are not enumerated."
                )
        else:
            raise Exception(
                f"LabelMismatch - The number of labels in the parameters `source` and `target` don't match."
            )

    else:
        for label in source:
            if label not in source_labels:
                raise Exception(
                    f"LabelNotAvailable - A label with the specified name does not exist."
                )
        data_rename[dimension] = target_values

    return data_rename
