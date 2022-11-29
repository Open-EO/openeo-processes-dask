import xarray as xr
from openeo_pg_parser_networkx.pg_schema import *

from openeo_processes_dask.exceptions import DimensionNotAvailable
from openeo_processes_dask.process_implementations.data_model import RasterCube

__all__ = ["create_raster_cube", "drop_dimension", "dimension_labels"]


def drop_dimension(data: RasterCube, dimension: str) -> RasterCube:
    if dimension not in data.dims:
        raise DimensionNotAvailable(
            "A dimension with the specified name does not exist."
        )

    return data.drop_dims(dimension)


def create_raster_cube() -> RasterCube:
    return xr.DataArray()


def dimension_labels(data: RasterCube, dimension: str) -> RasterCube:
    if dimension not in data.dims:
        raise DimensionNotAvailable(
            "A dimension with the specified name does not exist."
        )
    return data.coords[dimension]
