from __future__ import annotations

from typing import TypeVar

import dask_geopandas
import geopandas as gpd
import xarray as xr

RasterCube = TypeVar("RasterCube", "xr.DataArray", "xr.Dataset")
VectorCube = TypeVar("VectorCube", "gpd.GeoDataFrame", "dask_geopandas.GeoDataFrame")
