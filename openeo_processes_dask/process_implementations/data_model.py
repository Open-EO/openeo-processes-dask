from __future__ import annotations

from typing import TypeVar
import xarray as xr
import geopandas as gpd
import dask_geopandas


RasterCube = TypeVar("RasterCube", "xr.DataArray", "xr.Dataset")
VectorCube = TypeVar("VectorCube", "gpd.GeoDataFrame", "dask_geopandas.GeoDataFrame")
