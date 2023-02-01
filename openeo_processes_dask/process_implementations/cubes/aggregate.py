from typing import Callable, Optional, Union

import rasterio
from openeo_pg_parser_networkx.pg_schema import TemporalInterval, TemporalIntervals

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

    return resampled_data.reduce(reducer, keep_attrs=True)


def aggregate_spatial(
    data: RasterCube,
    geometries: VectorCube,
    reducer: Callable,
    target_dimension: str = "result",
    **kwargs,
) -> VectorCube:
    raise NotImplementedError
