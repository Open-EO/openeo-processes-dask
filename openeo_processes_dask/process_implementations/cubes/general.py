from typing import Optional

import xarray as xr
from openeo_pg_parser_networkx.pg_schema import *

from openeo_processes_dask.process_implementations.data_model import RasterCube
from openeo_processes_dask.process_implementations.exceptions import (
    DimensionLabelCountMismatch,
    DimensionNotAvailable,
)

__all__ = ["create_raster_cube", "drop_dimension", "dimension_labels", "add_dimension"]


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


def create_raster_cube() -> RasterCube:
    return xr.DataArray()


def dimension_labels(data: RasterCube, dimension: str) -> RasterCube:
    if dimension not in data.dims:
        raise DimensionNotAvailable(
            f"Provided dimension ({dimension}) not found in data.dims: {data.dims}"
        )
    return data.coords[dimension]


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
