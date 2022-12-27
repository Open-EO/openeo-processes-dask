from functools import partial
from typing import Callable, Optional

import dask
import dask_geopandas
import numpy as np
import rasterio
import xarray as xr
from datacube.utils.geometry import Geometry

from openeo_processes_dask.exceptions import DimensionNotAvailable, TooManyDimensions
from openeo_processes_dask.process_implementations.data_model import (
    RasterCube,
    VectorCube,
)

__all__ = ["aggregate_temporal", "aggregate_temporal_period", "aggregate_spatial"]


def geometry_mask(geoms, geobox, all_touched=False, invert=False):
    return rasterio.features.geometry_mask(
        [geom.to_crs(geobox.crs) for geom in geoms],
        out_shape=geobox.shape,
        transform=geobox.affine,
        all_touched=all_touched,
        invert=invert,
    )


def aggregate_temporal(
    data: RasterCube, intervals: list, reducer: Callable, **kwargs
) -> RasterCube:
    if "dimension" not in kwargs:
        kwargs["dimension"] = data.openeo.time_dim
    if not "labels" in kwargs:
        kwargs["labels"] = np.arange(len(kwargs["intervals"]))
    intervals = np.array(kwargs["intervals"])

    i = 0
    labels = [kwargs["labels"][0]]
    for label in kwargs["labels"][1:]:
        if i + 1 >= len(intervals) or intervals[i][1] == intervals[i + 1][0]:
            labels.append(label)
        else:
            labels.append("drop")
            labels.append(label)
        i += 1
    intervals = intervals.reshape(-1)
    intervals = np.unique(intervals)
    data = data.groupby_bins(group=kwargs["dimension"], bins=intervals, labels=labels)
    time_data = reducer(data)
    return time_data.sel(time_bins=kwargs["labels"])


def aggregate_temporal_period(
    data: RasterCube,
    reducer: Callable,
    period: str,
    dimension: Optional[str] = None,
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
    return reducer(data=resampled_data, **kwargs)


def aggregate_spatial(
    data: RasterCube,
    geometries: VectorCube,
    reducer: Callable,
    target_dimension: str = "result",
    **kwargs,
) -> VectorCube:
    if len(data.dims) > 3:
        raise TooManyDimensions(
            f"The number of dimensions must be reduced to three for aggregate_spatial. Input raster-cube dimensions: {data.dims}"
        )

    geometries = geometries.to_crs(data.rio.crs)
    geometries = geometries.reset_index(drop=True)

    crop_list = []
    valid_count_list = []
    total_count_list = []

    for _, row in geometries.iterrows():
        mask = geometry_mask(
            [Geometry(row["geometry"], crs=geometries.crs)], data.geobox, invert=True
        )
        xr_mask = xr.DataArray(
            mask,
            coords=[data.coords[data.openeo.y_dim], data.coords[data.openeo.x_dim]],
        )
        geom_crop = data.where(xr_mask).drop(["spatial_ref"], errors="ignore")
        crop_list.append(geom_crop)

        total_count = geom_crop.count(dim=data.openeo.spatial_dims)
        valid_count = geom_crop.where(~geom_crop.isnull()).count(
            dim=data.openeo.spatial_dims
        )
        valid_count_list.append(valid_count)
        total_count_list.append(total_count)

    try:
        reducer = partial(reducer, ignore_nodata=True)
    except TypeError:
        pass

    xr_crop_list = xr.concat(crop_list, "result")
    xr_crop_list_reduced = reducer(
        data=reducer(data=xr_crop_list, dimension=data.openeo.x_dim),
        dimension=data.openeo.y_dim,
    )
    xr_crop_list_reduced_ddf = (
        xr_crop_list_reduced.to_dataset(dim="bands")
        .to_dask_dataframe()
        .drop("result", axis=1)
    )
    output_ddf_merged = xr_crop_list_reduced_ddf.merge(geometries)

    valid_count_list_xr = xr.concat(valid_count_list, dim="valid_count")
    total_count_list_xr = xr.concat(total_count_list, dim="total_count")
    valid_count_list_xr_ddf = (
        valid_count_list_xr.to_dataset(dim="bands")
        .to_dask_dataframe()
        .drop("valid_count", axis=1)
        .add_suffix("_valid_count")
    )
    total_count_list_xr_ddf = (
        total_count_list_xr.to_dataset(dim="bands")
        .to_dask_dataframe()
        .drop("total_count", axis=1)
        .add_suffix("_total_count")
    )

    output_vector_cube = dask.dataframe.concat(
        [output_ddf_merged, valid_count_list_xr_ddf, total_count_list_xr_ddf], axis=1
    )

    output_vector_cube_ddf = dask_geopandas.from_dask_dataframe(output_vector_cube)
    output_vector_cube_ddf = output_vector_cube_ddf.set_crs(data.rio.crs)

    return output_vector_cube_ddf
