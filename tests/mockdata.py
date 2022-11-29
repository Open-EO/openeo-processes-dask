import logging

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import BoundingBox, TemporalInterval

logger = logging.getLogger(__name__)
rng = np.random.default_rng()


def generate_fake_rastercube(
    seed,
    spatial_extent: BoundingBox,
    temporal_extent: TemporalInterval,
    bands: list,
    n_spatial_coords: int,
    n_timesteps: int,
):

    # Calculate the desired resolution based on how many samples we desire on the longest axis.
    len_x = max(spatial_extent.west, spatial_extent.east) - min(
        spatial_extent.west, spatial_extent.east
    )
    len_y = max(spatial_extent.south, spatial_extent.north) - min(
        spatial_extent.south, spatial_extent.north
    )

    step_size = max(len_x, len_y) / n_spatial_coords

    x = np.arange(
        min(spatial_extent.west, spatial_extent.east),
        max(spatial_extent.west, spatial_extent.east),
        step=step_size,
    )
    y = np.arange(
        min(spatial_extent.south, spatial_extent.north),
        max(spatial_extent.south, spatial_extent.north),
        step=step_size,
    )
    t = pd.date_range(
        start=np.datetime64(temporal_extent.__root__[0].__root__),
        end=np.datetime64(temporal_extent.__root__[1].__root__),
        periods=n_timesteps,
    ).values

    coords = {"x": x, "y": y, "t": t, "bands": bands}

    # This is to enable simulating fake data from different collections.
    # The [:9] part is necessary because Dask.random.seed can only accept 32-bit values
    da.random.seed(int(str(abs(hash(seed)))[:9]))
    _data = da.random.random(tuple([len(v) for _, v in coords.items()]))

    data = xr.DataArray(
        data=_data,
        coords=coords,
        attrs={"crs": spatial_extent.crs},
    )
    data.rio.write_crs(spatial_extent.crs, inplace=True)

    return data.chunk("auto", "auto", "auto", -1)
