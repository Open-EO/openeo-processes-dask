import logging
import numpy as np
import pandas as pd
import dask.array as da
import xarray as xr
import pyproj

from openeo_pg_parser_networkx.pg_schema import BoundingBox, TemporalInterval

logger = logging.getLogger(__name__)
rng = np.random.default_rng()


def generate_fake_rastercube(seed, spatial_extent: BoundingBox, temporal_extent: TemporalInterval, bands: list, n_spatial_coords: int, n_timesteps: int):

    # Calculate the desired resolution based on how many samples we desire on the longest axis.
    len_x = max(spatial_extent.west, spatial_extent.east) - min(spatial_extent.west, spatial_extent.east)
    len_y = max(spatial_extent.south, spatial_extent.north) - min(spatial_extent.south, spatial_extent.north)
    
    step_size = max(len_x, len_y) / n_spatial_coords
    
    x = np.arange(min(spatial_extent.west, spatial_extent.east), max(spatial_extent.west, spatial_extent.east), step=step_size)
    y = np.arange(min(spatial_extent.south, spatial_extent.north), max(spatial_extent.south, spatial_extent.north), step=step_size)
    t = pd.date_range(start=np.datetime64(temporal_extent.__root__[0].__root__), end=np.datetime64(temporal_extent.__root__[1].__root__),
                  periods=n_timesteps).values

    coords = {"x": x, "y": y, "t": t, "bands": bands}
    
    da.random.seed(int(str(abs(hash(seed)))[:9]))
    _data = da.random.random(tuple([len(v) for _, v in coords.items()]))

    data = xr.DataArray(
        data=_data,
        coords=coords,
        attrs={"crs": spatial_extent.crs,
                "origin": 'test'},
    )
    data.rio.write_crs(spatial_extent.crs, inplace=True)

    return data.chunk("auto", "auto", "auto", -1)

def generate_fake_bounding_box(crs: pyproj.CRS):
    # if crs.axis_info[0].unit_name == "degrees":
    #     size = 
    # elif crs.axis_info[0].unit_name == "metres":
    #     size = 
    
    random_x = rng.uniform(-1, 0, 1)
    random_y = np.random.RandomState()


    return BoundingBox(west=crs.area_of_use.west, east=crs.area_of_use.east, north=crs.area_of_use.north, south=crs.area_of_use.south, crs=crs)
    


def generate_fake_temporal_interval():
    pass
