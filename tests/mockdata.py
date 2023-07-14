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
            start=np.datetime64(temporal_extent.__root__[0].__root__),
            end=np.datetime64(temporal_extent.__root__[1].__root__),
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
