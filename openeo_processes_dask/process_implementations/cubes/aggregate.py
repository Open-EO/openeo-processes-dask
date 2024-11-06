import copy
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

    intervals_np = (
        np.array(intervals, dtype=np.datetime64).astype("datetime64[s]").astype(float)
    )
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
    data_copy = copy.deepcopy(data)
    t_coords = data_copy[t].values.astype(str)
    data_copy[t] = np.array(t_coords, dtype="datetime64[s]").astype(float)
    grouped_data = data_copy.groupby_bins(t, bins=intervals_flat)
    positional_parameters = {"data": 0}
    groups = grouped_data.reduce(
        reducer, keep_attrs=True, positional_parameters=positional_parameters
    )
    groups[t + "_bins"] = labels_nans
    data_agg_temp = groups.sel({t + "_bins": labels})
    data_agg_temp = data_agg_temp.rename({t + "_bins": t})

    return data_agg_temp


def get_intervals(data, period):
    format = "%Y-%m-%dT%H:%M:%S"
    start, end = data["t"].values[0], data["t"].values[-1]
    year_start = pd.to_datetime(start).year
    year_end = pd.to_datetime(end).year
    month_start = pd.to_datetime(start).month
    month_end = pd.to_datetime(end).month

    if period == "decade":
        year_start = np.datetime64(
            (np.floor(year_start / 10) * 10).astype(int).astype(str)
        )
        year_end = np.datetime64((np.ceil(year_end / 10) * 10).astype(int).astype(str))
        intervals = pd.date_range(start=year_start, end=year_end, freq="10YS").strftime(
            format
        )
        labels = pd.date_range(start=year_start, end=year_end, freq="10YS").strftime(
            "%Y"
        )[:-1]
    elif period == "decade-ad":
        year_start = np.datetime64(
            (np.floor(year_start / 10) * 10 + 1).astype(int).astype(str)
        )
        year_end = np.datetime64(
            (np.ceil(year_end / 10) * 10 + 1).astype(int).astype(str)
        )
        intervals = pd.date_range(start=year_start, end=year_end, freq="10YS").strftime(
            format
        )
        labels = pd.date_range(start=year_start, end=year_end, freq="10YS").strftime(
            "%Y"
        )[:-1]
    elif period == "tropical-season":
        if month_start >= 5 and month_start < 10:
            month_start = np.datetime64(str(year_start) + "-05-01")
        elif month_start < 5:
            month_start = np.datetime64(str(year_start - 1) + "-11-01")
        else:
            month_start = np.datetime64(str(year_start) + "-11-01")
        if month_end >= 5 and month_end < 10:
            month_end = np.datetime64(str(year_end) + "-11-01")
        elif month_end < 5:
            month_end = np.datetime64(str(year_end) + "-05-01")
        else:
            month_end = np.datetime64(str(year_end + 1) + "-05-01")
        intervals = pd.period_range(
            start=month_start, end=month_end, freq="6M"
        ).strftime(format)
        labels = []
        for interval in intervals[:-1]:
            if "-11-" in interval:
                labels.append(interval[:5] + "ndjfma")
            if "-05-" in interval:
                labels.append(interval[:5] + "mjjaso")
    elif period == "dekad":
        day = pd.to_datetime(start).day
        day_start = (np.floor(day / 10) * 10 + 1).astype(int).astype(str)
        day_start = f"{year_start}-{month_start}-{day_start}"
        intervals = pd.date_range(
            start=day_start, end=f"{year_start}-{month_start}-22", freq="10D"
        ).strftime(format)
        for date in pd.date_range(
            start=f"{year_start}-{month_start}-22", end=end, freq="1MS"
        )[:-1]:
            intervals = intervals.append(
                pd.date_range(start=date, freq="10D", periods=3).strftime(format)
            )
        day = pd.to_datetime(end).day
        periods = (np.ceil((day - 1) / 10)).astype(int)
        day_end = f"{year_end}-{month_end}-01"
        if day > 21:
            day_end = pd.date_range(start=day_end, freq="10D", periods=periods)
            days = 7
            last_day = day_end[-1] + pd.DateOffset(days=7)
            while last_day.day != 1:
                days += 1
                last_day = day_end[-1] + pd.DateOffset(days=days)
            day_end = day_end.append(pd.DatetimeIndex(data=[str(last_day)])).strftime(
                format
            )
        else:
            day_end = pd.date_range(
                start=day_end, freq="10D", periods=periods + 1
            ).strftime(format)
        intervals = intervals.append(day_end)
        labels = []
        for interval in intervals[:-1]:
            year = pd.DatetimeIndex(data=[interval]).year.astype(int)[0]
            dekad = int(pd.DatetimeIndex(data=[interval]).day_of_year[0] / 10)
            label = f"{year}-{dekad}" if dekad > 9 else f"{year}-0{dekad}"
            labels.append(label)
    else:
        raise NotImplementedError(
            f"The provided period '{period})' is not implemented. "
        )
    interval_array = np.array(intervals, dtype=str)
    interval_matrix = np.zeros((len(interval_array) - 1, 2)).astype(str)
    interval_matrix[:, 0] = interval_array[:-1]
    interval_matrix[:, 1] = interval_array[1:]
    return interval_matrix, list(labels)


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
        resampled_data = data.resample({applicable_temporal_dimension: frequency})

        positional_parameters = {"data": 0}
        return resampled_data.reduce(
            reducer, keep_attrs=True, positional_parameters=positional_parameters
        )

    else:
        intervals, labels = get_intervals(data, period)
        return aggregate_temporal(
            data=data, intervals=intervals, reducer=reducer, labels=labels
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
