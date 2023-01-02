from typing import Callable

from openeo_pg_parser_networkx.pg_schema import BoundingBox, GeoJson, TemporalInterval

from openeo_processes_dask.exceptions import DimensionNotAvailable
from openeo_processes_dask.process_implementations.data_model import RasterCube


def filter_spatial(data: RasterCube, geometries: GeoJson, **kwargs) -> RasterCube:
    raise NotImplementedError()


def filter_bbox(data: RasterCube, extent: BoundingBox) -> RasterCube:
    raise NotImplementedError()


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
