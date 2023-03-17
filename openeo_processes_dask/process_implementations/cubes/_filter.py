import logging
from typing import Callable

import numpy as np
from openeo_pg_parser_networkx.pg_schema import BoundingBox, GeoJson, TemporalInterval

from openeo_processes_dask.process_implementations.data_model import RasterCube
from openeo_processes_dask.process_implementations.exceptions import (
    DimensionNotAvailable,
    TooManyDimensions,
)

logger = logging.getLogger(__name__)


def filter_spatial(data: RasterCube, geometries: GeoJson, **kwargs) -> RasterCube:
    raise NotImplementedError()


def filter_bbox(data: RasterCube, extent: BoundingBox) -> RasterCube:
    raise NotImplementedError()


def filter_temporal(
    data: RasterCube, extent: TemporalInterval, dimension: str = None
) -> RasterCube:
    temporal_dims = data.openeo.temporal_dims

    if dimension is not None:
        if dimension not in data.dims:
            raise DimensionNotAvailable(
                f"A dimension with the specified name: {dimension} does not exist."
            )
        applicable_temporal_dimension = dimension
        if dimension not in temporal_dims:
            logger.warning(
                f"The selected dimension {dimension} exists but it is not labeled as a temporal dimension. Available temporal diemnsions are {temporal_dims}."
            )
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

    start_time = extent[0]
    if start_time is not None:
        start_time = start_time.to_numpy()
    end_time = extent[1]
    if end_time is not None:
        end_time = extent[1].to_numpy() - np.timedelta64(1, "ms")
    # The second element is the end of the temporal interval.
    # The specified instance in time is excluded from the interval.
    # See https://processes.openeo.org/#filter_temporal

    filtered = data.loc[{applicable_temporal_dimension: slice(start_time, end_time)}]

    return filtered


def filter_labels(
    data: RasterCube, condition: Callable, dimension: str, **kwargs
) -> RasterCube:
    if dimension not in data.dims:
        raise DimensionNotAvailable(
            f"Provided dimension ({dimension}) not found in data.dims: {data.dims}"
        )

    labels = data[dimension].values
    label_mask = condition(x=labels)
    label = labels[label_mask]
    data = data.sel(**{dimension: label})
    return data
