import xarray as xr
from openeo_pg_parser_networkx.pg_schema import *

from openeo_processes_dask.exceptions import (
    DimensionLabelCountMismatch,
    DimensionNotAvailable,
)
from openeo_processes_dask.process_implementations.data_model import RasterCube

__all__ = ["create_raster_cube", "drop_dimension", "dimension_labels"]


def drop_dimension(data: RasterCube, name: str) -> RasterCube:
    if name not in data.dims:
        raise DimensionNotAvailable()
    if len(data[name]) > 1:
        raise DimensionLabelCountMismatch()
    return data.drop(name)


def create_raster_cube() -> RasterCube:
    return xr.DataArray()


def dimension_labels(data: RasterCube, dimension: str) -> RasterCube:
    if dimension not in data.dims:
        raise DimensionNotAvailable()
    return data.coords[dimension]
