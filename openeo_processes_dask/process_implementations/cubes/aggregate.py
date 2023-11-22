import gc
import logging
from typing import Callable, Optional, Union

import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
import xarray as xr
import xvec
from joblib import Parallel, delayed
from openeo_pg_parser_networkx.pg_schema import TemporalInterval, TemporalIntervals

from openeo_processes_dask.process_implementations.data_model import (
    RasterCube,
    VectorCube,
)
from openeo_processes_dask.process_implementations.exceptions import (
    DimensionNotAvailable,
    TooManyDimensions,
)

DEFAULT_CRS = "EPSG:4326"


__all__ = ["aggregate_temporal", "aggregate_temporal_period", "aggregate_spatial"]

logger = logging.getLogger(__name__)


def geometry_mask(geoms, geobox, all_touched=False, invert=False):
    return rasterio.features.geometry_mask(
        [geom.to_crs(geobox.crs) for geom in geoms],
        out_shape=geobox.shape,
        transform=geobox.affine,
        all_touched=all_touched,
        invert=invert,
    )


def aggregate_temporal(
    data: RasterCube,
    intervals: Union[TemporalIntervals, list[TemporalInterval], list[Optional[str]]],
    reducer: Callable,
    labels: Optional[list] = None,
    dimension: Optional[str] = None,
    context: Optional[dict] = None,
    **kwargs,
) -> RasterCube:
    temporal_dims = data.openeo.temporal_dims

    if dimension is not None:
        if dimension not in data.dims:
            raise DimensionNotAvailable(
                f"A dimension with the specified name: {dimension} does not exist."
            )
        applicable_temporal_dimension = dimension
    else:
        if not temporal_dims:
            raise DimensionNotAvailable(
                f"No temporal dimension detected on dataset. Available dimensions: {data.dims}"
            )
        if len(temporal_dims) > 1:
            raise TooManyDimensions(
                f"The data cube contains multiple temporal dimensions: {temporal_dims}. The parameter `dimension` must be specified."
            )
        applicable_temporal_dimension = temporal_dims[0]

    aggregated_data = data.groupby_bins(
        group=applicable_temporal_dimension, labels=labels
    )

    raise NotImplementedError("aggregate_temporal is currently not implemented")


def aggregate_temporal_period(
    data: RasterCube,
    reducer: Callable,
    period: str,
    dimension: Optional[str] = None,
) -> RasterCube:
    temporal_dims = data.openeo.temporal_dims

    if dimension is not None:
        if dimension not in data.dims:
            raise DimensionNotAvailable(
                f"A dimension with the specified name: {dimension} does not exist."
            )
        applicable_temporal_dimension = dimension
    else:
        if not temporal_dims:
            raise DimensionNotAvailable(
                f"No temporal dimension detected on dataset. Available dimensions: {data.dims}"
            )
        if len(temporal_dims) > 1:
            raise TooManyDimensions(
                f"The data cube contains multiple temporal dimensions: {temporal_dims}. The parameter `dimension` must be specified."
            )
        applicable_temporal_dimension = temporal_dims[0]

    periods_to_frequency = {
        "hour": "H",
        "day": "D",
        "week": "W",
        "month": "M",
        "season": "QS-DEC",
        "year": "AS",
    }

    if period in periods_to_frequency.keys():
        frequency = periods_to_frequency[period]
    else:
        raise NotImplementedError(
            f"The provided period '{period})' is not implemented yet. The available ones are {list(periods_to_frequency.keys())}."
        )

    resampled_data = data.resample({applicable_temporal_dimension: frequency})

    positional_parameters = {"data": 0}
    return resampled_data.reduce(
        reducer, keep_attrs=True, positional_parameters=positional_parameters
    )


def _aggregate_geometry(
    data: RasterCube,
    geom,
    transform,
    reducer: Callable,
):
    data_dims = list(data.dims)
    y_dim = data.openeo.y_dim
    x_dim = data.openeo.x_dim
    t_dim = data.openeo.temporal_dims
    t_dim = None if len(t_dim) == 0 else t_dim[0]
    b_dim = data.openeo.band_dims
    b_dim = None if len(b_dim) == 0 else b_dim[0]

    y_dim_size = data.sizes[y_dim]
    x_dim_size = data.sizes[x_dim]

    # Create a GeoSeries from the geometry
    geo_series = gpd.GeoSeries(geom)

    # Convert the GeoSeries to a GeometryArray
    geometry_array = geo_series.geometry.array

    mask = rasterio.features.geometry_mask(
        geometry_array, out_shape=(y_dim_size, x_dim_size), transform=transform
    )

    if t_dim is not None:
        mask = np.expand_dims(mask, axis=data_dims.index(t_dim))
    if b_dim is not None:
        mask = np.expand_dims(mask, axis=data_dims.index(b_dim))

    masked_data = data * mask
    del mask, data
    gc.collect()

    positional_parameters = {"data": 0}

    stat_within_polygon = masked_data.reduce(
        reducer,
        axis=(data_dims.index(y_dim), data_dims.index(x_dim)),
        keep_attrs=True,
        ignore_nodata=True,
        positional_parameters=positional_parameters,
    )
    result = stat_within_polygon.values

    del masked_data, stat_within_polygon
    gc.collect()

    return result.T


def aggregate_spatial(
    data: RasterCube,
    geometries,
    reducer: Callable,
    chunk_size: int = 2,
) -> VectorCube:
    t_dim = data.openeo.temporal_dims
    t_dim = None if len(t_dim) == 0 else t_dim[0]
    b_dim = data.openeo.band_dims
    b_dim = None if len(b_dim) == 0 else b_dim[0]

    if "type" in geometries and geometries["type"] == "FeatureCollection":
        gdf = gpd.GeoDataFrame.from_features(geometries, DEFAULT_CRS)
    elif "type" in geometries and geometries["type"] in ["Polygon"]:
        polygon = shapely.geometry.Polygon(geometries["coordinates"][0])
        gdf = gpd.GeoDataFrame(geometry=[polygon])
        gdf.crs = DEFAULT_CRS

    transform = data.rio.transform()
    geometries = gdf.geometry.values

    geometry_chunks = [
        geometries[i : i + chunk_size] for i in range(0, len(geometries), chunk_size)
    ]

    computed_results = []
    logger.info(f"Running aggregate_spatial process")
    try:
        for i, chunk in enumerate(geometry_chunks):
            # Create a list of delayed objects for the current chunk
            chunk_results = Parallel(n_jobs=-1)(
                delayed(_aggregate_geometry)(
                    data, geom, transform=transform, reducer=reducer
                )
                for geom in chunk
            )
            computed_results.extend(chunk_results)
    except:
        logger.debug(f"Running process failed at {(i+1) *2} geometry")

    logger.info(f"Finish aggregate_spatial process for {len(geometries)}")

    final_results = np.stack(computed_results)
    del chunk_results, geometry_chunks, computed_results
    gc.collect()

    df = pd.DataFrame()
    keys_items = {}

    for idx, b in enumerate(data[b_dim].values):
        columns = []
        if t_dim:
            for t in range(len(data[t_dim])):
                columns.append(f"{b}_time{t+1}")
            aggregated_data = final_results[:, idx, :]
        else:
            columns.append(f"{b}")
            aggregated_data = final_results[:, idx]

        keys_items[b] = columns

        # Create a new DataFrame with the current data and columns
        band_df = pd.DataFrame(aggregated_data, columns=columns)
        # Concatenate the new DataFrame with the existing DataFrame
        df = pd.concat([df, band_df], axis=1)

    df = gpd.GeoDataFrame(df, geometry=gdf.geometry)

    data_vars = {}
    for key in keys_items.keys():
        data_vars[key] = (["geometry", t_dim], df[keys_items[key]])

    ## Create VectorCube
    if t_dim:
        times = list(data[t_dim].values)
        vec_cube = xr.Dataset(
            data_vars=data_vars, coords={"geometry": df.geometry, t_dim: times}
        ).xvec.set_geom_indexes("geometry", crs=df.crs)
    else:
        vec_cube = xr.Dataset(
            data_vars=data_vars, coords=dict(geometry=df.geometry)
        ).xvec.set_geom_indexes("geometry", crs=df.crs)

    return vec_cube
