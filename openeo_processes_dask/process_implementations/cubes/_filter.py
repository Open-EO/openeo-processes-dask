import logging
from typing import Callable

from openeo_pg_parser_networkx.pg_schema import BoundingBox, GeoJson, TemporalInterval

from openeo_processes_dask.process_implementations.data_model import RasterCube
from openeo_processes_dask.process_implementations.exceptions import (
    DimensionNotAvailable,
)

logger = logging.getLogger(__name__)

__all__ = ["filter_labels"]


def filter_temporal(
    data: RasterCube, extent: TemporalInterval, dimension: str
) -> RasterCube:
    raise NotImplementedError()


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
