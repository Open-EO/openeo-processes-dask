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

__all__ = ["aggregate_temporal", "aggregate_temporal_period", "aggregate_spatial"]

logger = logging.getLogger(__name__)


def aggregate_temporal(
    data: RasterCube,
    intervals: Union[TemporalIntervals, list[TemporalInterval], list[Optional[str]]],
    reducer: Callable,
    labels: Optional[list] = None,
    dimension: Optional[str] = None,
    context: Optional[dict] = None,
    **kwargs,
) -> RasterCube:
    from xarray.groupers import BinGrouper

    temporal_dims = data.openeo.temporal_dims

    if dimension is not None:
        if dimension not in data.dims:
            raise DimensionNotAvailable(
                f"A dimension with the specified name: {dimension} does not exist."
            )
        t = dimension
    else:
        if not temporal_dims:
            raise DimensionNotAvailable(
                f"No temporal dimension detected on dataset. Available dimensions: {data.dims}"
            )
        if len(temporal_dims) > 1:
            raise TooManyDimensions(
                f"The data cube contains multiple temporal dimensions: {temporal_dims}. The parameter `dimension` must be specified."
            )
        t = temporal_dims[0]

    intervals_np = np.array(intervals, dtype=np.datetime64).astype("float")
    intervals_flat = np.reshape(
        intervals_np, np.shape(intervals_np)[0] * np.shape(intervals_np)[1]
    )

    if not labels:
        labels = np.array(intervals, dtype="datetime64[s]").astype(str)[:, 0]
    if (intervals_np[1:, 0] < intervals_np[:-1, 1]).any():
        raise NotImplementedError(
            "Aggregating data for overlapping time ranges is not implemented. "
        )

    mask = np.zeros((len(labels) * 2) - 2).astype(bool)
    mask[1::2] = np.isin(intervals_np[1:, 0], intervals_np[:-1, 1])
    mask = np.append(mask, np.array([False, True]))

    labels_nans = np.arange(len(labels) * 2).astype(str)
    labels_nans[::2] = labels
    labels_nans = labels_nans[~mask]

    intervals_flat = np.unique(intervals_flat)
    data[t] = data[t].values.astype("float")
    grouped_data = data.groupby({t: BinGrouper(bins=intervals_flat)})
    positional_parameters = {"data": 0}
    groups = grouped_data.reduce(
        reducer, keep_attrs=True, positional_parameters=positional_parameters
    )
    groups[t + "_bins"] = labels_nans
    data_agg_temp = groups.sel({t + "_bins": labels})
    data_agg_temp = data_agg_temp.rename(t + "_bins", t)

    return data_agg_temp


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


def aggregate_spatial(
    data: RasterCube,
    geometries,
    reducer: Callable,
    chunk_size: int = 2,
) -> VectorCube:
    x_dim = data.openeo.x_dim
    y_dim = data.openeo.y_dim
    DEFAULT_CRS = "EPSG:4326"

    if isinstance(geometries, str):
        # Allow importing geometries from url (e.g. github raw)
        import json
        from urllib.request import urlopen

        response = urlopen(geometries)
        geometries = json.loads(response.read())
    if isinstance(geometries, dict):
        # Get crs from geometries
        if "features" in geometries:
            for feature in geometries["features"]:
                if "properties" not in feature:
                    feature["properties"] = {}
                elif feature["properties"] is None:
                    feature["properties"] = {}
            if isinstance(geometries.get("crs", {}), dict):
                DEFAULT_CRS = (
                    geometries.get("crs", {})
                    .get("properties", {})
                    .get("name", DEFAULT_CRS)
                )
            else:
                DEFAULT_CRS = int(geometries.get("crs", {}))
            logger.info(f"CRS in geometries: {DEFAULT_CRS}.")

        if "type" in geometries and geometries["type"] == "FeatureCollection":
            gdf = gpd.GeoDataFrame.from_features(geometries, crs=DEFAULT_CRS)
        elif "type" in geometries and geometries["type"] in ["Polygon"]:
            polygon = shapely.geometry.Polygon(geometries["coordinates"][0])
            gdf = gpd.GeoDataFrame(geometry=[polygon])
            gdf.crs = DEFAULT_CRS

    if isinstance(geometries, xr.Dataset):
        if hasattr(geometries, "xvec"):
            gdf = geometries.xvec.to_geodataframe()

    if isinstance(geometries, gpd.GeoDataFrame):
        gdf = geometries

    gdf = gdf.to_crs(data.rio.crs)
    geometries = gdf.geometry.values

    positional_parameters = {"data": 0}
    vec_cube = data.xvec.zonal_stats(
        geometries,
        x_coords=x_dim,
        y_coords=y_dim,
        method="iterate",
        stats=reducer,
        positional_parameters=positional_parameters,
    )
    return vec_cube
