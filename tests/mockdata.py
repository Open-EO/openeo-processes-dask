import logging
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import BoundingBox, TemporalInterval

logger = logging.getLogger(__name__)


def create_fake_rastercube(
    data,
    spatial_extent: BoundingBox,
    temporal_extent: TemporalInterval,
    bands: list,
    backend="numpy",
    chunks=("auto", "auto", "auto", -1),
):
    # Calculate the desired resolution based on how many samples we desire on the longest axis.
    len_x = max(spatial_extent.west, spatial_extent.east) - min(
        spatial_extent.west, spatial_extent.east
    )
    len_y = max(spatial_extent.south, spatial_extent.north) - min(
        spatial_extent.south, spatial_extent.north
    )

    x_coords = np.arange(
        min(spatial_extent.west, spatial_extent.east),
        max(spatial_extent.west, spatial_extent.east),
        step=len_x / data.shape[0],
    )
    y_coords = np.arange(
        min(spatial_extent.south, spatial_extent.north),
        max(spatial_extent.south, spatial_extent.north),
        step=len_y / data.shape[1],
    )

    # This line raises a deprecation warning, which according to this thread
    # will never actually be deprecated:
    # https://github.com/numpy/numpy/issues/23904
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        t_coords = pd.date_range(
            start=np.datetime64(temporal_extent.root[0].root),
            end=np.datetime64(temporal_extent.root[1].root),
            periods=data.shape[2],
        ).values

    coords = {"x": x_coords, "y": y_coords, "t": t_coords, "bands": bands}

    raster_cube = xr.DataArray(
        data=data,
        coords=coords,
        attrs={"crs": spatial_extent.crs},
    )
    raster_cube.rio.write_crs(spatial_extent.crs, inplace=True)

    if "dask" in backend:
        import dask.array as da

        raster_cube.data = da.from_array(raster_cube.data, chunks=chunks)

    return raster_cube


def create_fake_curvilinear_rastercube(
    data,
    spatial_extent: BoundingBox,
    temporal_extent: TemporalInterval,
    bands: list,
    backend="numpy",
    chunks=("auto", "auto", "auto", -1),
    lon_name: str = "lon",
    lat_name: str = "lat",
    warp_strength: float = 0.02,
):
    """Create a fake *curvilinear* cube.

    This matches what resample_spatial(method="geocode") expects:
      - regular index dims (y, x)
      - 2D lon/lat layers aligned to (y, x) as coords (or data vars)
      - payload in a DataArray with optional (t, bands)
    """

    # Base 1D lon/lat axes used only to build the 2D curvilinear lon/lat.
    len_x = max(spatial_extent.west, spatial_extent.east) - min(
        spatial_extent.west, spatial_extent.east
    )
    len_y = max(spatial_extent.south, spatial_extent.north) - min(
        spatial_extent.south, spatial_extent.north
    )

    nx = data.shape[0]
    ny = data.shape[1]

    lon_1d = np.linspace(
        min(spatial_extent.west, spatial_extent.east),
        max(spatial_extent.west, spatial_extent.east),
        num=nx,
        endpoint=False,
    )
    lat_1d = np.linspace(
        min(spatial_extent.south, spatial_extent.north),
        max(spatial_extent.south, spatial_extent.north),
        num=ny,
        endpoint=False,
    )

    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)

    # Make it curvilinear by adding a smooth warp that depends on both i and j.
    if len_x > 0:
        lon2d = lon2d + (warp_strength * len_x) * np.sin(
            2.0 * np.pi * (lat2d - lat2d.min()) / max(len_y, 1e-12)
        )
    if len_y > 0:
        lat2d = lat2d + (warp_strength * len_y) * np.cos(
            2.0 * np.pi * (lon2d - lon2d.min()) / max(len_x, 1e-12)
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        t_coords = pd.date_range(
            start=np.datetime64(temporal_extent.root[0].root),
            end=np.datetime64(temporal_extent.root[1].root),
            periods=data.shape[2],
        ).values

    # Use integer index coords for x/y; lon/lat are the real geo coords.
    x_idx = np.arange(nx)
    y_idx = np.arange(ny)

    coords = {
        "x": x_idx,
        "y": y_idx,
        "t": t_coords,
        "bands": bands,
        lon_name: (("y", "x"), lon2d.astype(np.float64)),
        lat_name: (("y", "x"), lat2d.astype(np.float64)),
    }

    raster_cube = xr.DataArray(data=data, dims=("x", "y", "t", "bands"), coords=coords)

    if "dask" in backend:
        import dask.array as da

        raster_cube.data = da.from_array(raster_cube.data, chunks=chunks)

    return raster_cube
